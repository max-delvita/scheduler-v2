import { BaseMessage } from '@langchain/core/messages';
import { StateGraph, END } from "@langchain/langgraph";
import { ChatOpenAI } from "@langchain/openai";
import { JsonOutputParser } from "@langchain/core/output_parsers";
import { PromptTemplate } from "@langchain/core/prompts";
import { Client as PostmarkClient } from "postmark"; // Import Postmark client
import { AIMessage, HumanMessage } from "@langchain/core/messages";
import { saveMessage } from "./db"; // Import saveMessage
import type { RunnableConfig } from "@langchain/core/runnables"; // Import config type

/**
 * Defines the structure for storing information
 * throughout the meeting scheduling process.
 */
export interface AgentState {
  // Email details from the initial request
  initial_sender_email: string | null;
  initial_sender_name: string | null;
  initial_subject: string | null;
  initial_body: string | null;
  initial_recipients?: string[] | null; // Raw To/Cc list from initial email

  // Meeting details
  organizer_email: string | null;
  organizer_availability: string[] | null; // Store proposed slots/constraints
  participants: string[] | null; // List of participant emails
  participant_availability: Record<string, string[]> | null; // Map email to their slots
  confirmed_datetime: Date | null;
  meeting_topic: string | null; // Extracted from subject/body

  // Agent's internal state
  current_step: string; // e.g., 'start', 'get_organizer_availability', 'get_participant_availability', 'confirming', 'done'
  error_message: string | null; // If something goes wrong
  email_history: BaseMessage[]; // Keep track of the conversation (using LangChain message types)

  // Internal tracking for saving messages
  last_agent_message_id?: string | null; 

  // Internal fields loaded from DB for specific actions
  _webhook_target_address?: string | null;
}

/**
 * Represents the core workflow/graph for the scheduling agent.
 */
const workflow = new StateGraph<AgentState>({
    channels: {
        // Define how state keys are updated. 
        // 'reducer' is common for complex updates, 
        // others like 'value' simply replace the value.
        initial_sender_email: { value: (x, y) => y ?? x },
        initial_sender_name: { value: (x, y) => y ?? x },
        initial_subject: { value: (x, y) => y ?? x },
        initial_body: { value: (x, y) => y ?? x },
        organizer_email: { value: (x, y) => y ?? x },
        organizer_availability: { value: (x, y) => y ?? x },
        participants: { value: (x, y) => y ?? x },
        participant_availability: { value: (x, y) => y ?? x },
        confirmed_datetime: { value: (x, y) => y ?? x },
        meeting_topic: { value: (x, y) => y ?? x },
        current_step: { value: (x, y) => y ?? x },
        error_message: { value: (x, y) => y ?? x },
        // For email_history, we want to append new messages, not replace
        email_history: {
            value: (x: BaseMessage[], y: BaseMessage[]) => x.concat(y),
            default: () => [],
        },
        last_agent_message_id: { value: (x, y) => y ?? x }, // Add channel for new field
        _webhook_target_address: { value: (x, y) => y ?? x }, // ADD THIS CHANNEL
        initial_recipients: { value: (x, y) => y ?? x }, // ADD THIS CHANNEL
    }
});

// --- Node Definitions ---

/**
 * The initial node that processes the first incoming email.
 */
async function handle_start(state: AgentState): Promise<Partial<AgentState>> {
    console.log("--- Agent Node: handle_start --- ");

    // If we are resuming a session (step is not 'start'), just pass through
    if (state.current_step && state.current_step !== 'start') {
        console.log(`Resuming session, skipping initial analysis. Current step: ${state.current_step}`);
        return {}; // Return no changes, router will handle next step
    }

    // --- Otherwise, perform initial analysis (existing logic) --- 
    console.log("Performing initial analysis for new session...");
    const parser = new JsonOutputParser<InitialAnalysis>();

    const prompt = PromptTemplate.fromTemplate(
`You are Amy, an expert meeting scheduling assistant for askSymple.ai.
Analyze the following email content and headers from a potential organizer ({sender}).

Email Details:
Sender: {sender}
To: {recipients_to}
Cc: {recipients_cc}
Subject: {subject}
Body:
{body}

Your goal is to understand:
1.  **Intent**: Is the primary intent to schedule a new meeting? Use "schedule_meeting" or "other".
2.  **Topic**: What is the meeting about? (e.g., "Project Alpha Sync")
3.  **Participants**: List all relevant participant email addresses. Consider recipients in the To/Cc fields (excluding the agent's own email, {agent_email}) AND any participants explicitly mentioned in the email body/subject. Combine these sources and list unique emails.
4.  **Proposed Times**: List any specific dates, times, or time ranges suggested by the organizer for the meeting.

**CRITICAL**: Respond ONLY with a single, valid JSON object containing ONLY the fields "intent", "topic", "participants", and "proposed_times" EXACTLY as shown in the format instructions. Do NOT add any other fields or nesting.

Format Instructions:
{format_instructions}

JSON Output:`
    );

    // Create a chain: Prompt -> LLM -> Parser
    const chain = prompt.pipe(llm).pipe(parser);

    console.log("Analyzing initial email with LLM...");
    try {
        const analysis: InitialAnalysis = await chain.invoke({
            sender: state.initial_sender_email,
            // Pass To/Cc info (might need refining based on how passed from index.ts)
            recipients_to: state.initial_recipients?.join(', ') ?? "", 
            recipients_cc: "", // Assuming Cc is less critical or combined earlier
            agent_email: process.env.POSTMARK_SENDER_EMAIL ?? "amy@asksymple.ai",
            subject: state.initial_subject ?? "",
            body: state.initial_body ?? "",
            format_instructions: parser.getFormatInstructions(),
        });

        console.log("LLM Analysis Result:", analysis);

        if (analysis.intent !== "schedule_meeting") {
            console.log("Intent is not to schedule a meeting. Ending flow.");
            // TODO: Maybe send a clarifying email?
            return { 
                current_step: "end_other_intent", // A new step to indicate why we stopped
                error_message: "User intent was not to schedule a meeting."
            };
        }

        // Determine next step based on analysis
        let next_step: string;
        let organizer_availability: string[] | null = null;

        if (analysis.proposed_times && analysis.proposed_times.length > 0) {
            console.log("Organizer proposed initial times.");
            organizer_availability = analysis.proposed_times;
            // If times proposed, check if we need to ask participants or confirm
            if (analysis.participants && analysis.participants.length > 0) {
                next_step = "get_participant_availability";
            } else {
                // No participants, just need to confirm time with organizer? 
                // Or maybe just finalize if only one time proposed?
                // Let's go to a confirm step for now.
                next_step = "confirm_time"; 
            }
        } else {
            // No times proposed, need to ask the organizer
            console.log("No initial times proposed by organizer.");
            next_step = "get_organizer_availability";
        }

        // Update state based on analysis and decided next step
        return {
            organizer_email: state.initial_sender_email,
            meeting_topic: analysis.topic,
            participants: analysis.participants,
            organizer_availability: organizer_availability, // Set if proposed
            current_step: next_step 
        };

    } catch (error) {
        console.error("Error analyzing email with LLM:", error);
        return {
            current_step: "error",
            error_message: "Failed to analyze initial email."
        };
    }
}

/**
 * Generates and sends an email to the organizer asking for their availability.
 */
async function ask_organizer_availability(state: AgentState): Promise<Partial<AgentState>> {
    console.log("--- Agent Node: ask_organizer_availability ---");

    if (!state.organizer_email) {
        console.error("Organizer email missing, cannot ask for availability.");
        return { current_step: "error", error_message: "Organizer email missing." };
    }

    // Use LLM to draft the email
    const draftPrompt = PromptTemplate.fromTemplate(
`You are Amy, an expert meeting scheduling assistant for askSymple.ai.
Your goal is to politely ask the meeting organizer ({organizer_email}) for their availability for a meeting about "{topic}".
Keep the email concise and professional. Ask for specific dates and times or general availability for the upcoming week or relevant timeframe.

Draft ONLY the body of the email. Do not include subject or greetings like "Hi [Name]".

Email Body Draft:`
    );
    
    const draftChain = draftPrompt.pipe(llm); // Just need the raw string output

    try {
        console.log("Drafting email to organizer...");
        const emailBodyDraft = await draftChain.invoke({
            organizer_email: state.organizer_email,
            topic: state.meeting_topic ?? "our meeting" // Fallback topic
        });

        console.log(`Generated Email Body Draft: ${emailBodyDraft.content}`);

        // Construct the full email content
        const subject = `Availability for ${state.meeting_topic ?? 'meeting'}`;
        const fullEmailBody = `Hi ${state.initial_sender_name ?? 'there'},

${emailBodyDraft.content}

Best regards,
Amy
askSymple.ai Scheduling Assistant`;

        if (!state._webhook_target_address) {
            console.error("Webhook target address missing from state, cannot set Reply-To header.");
            // Decide how to handle: error out, or send without Reply-To?
            // Let's error out for now to make the issue visible.
            return { current_step: "error", error_message: "Webhook target address missing for Reply-To." };
        }

        // Send email via Postmark
        console.log(`Sending availability request to ${state.organizer_email}...`);
        const sendResponse = await postmarkClient.sendEmail({
            From: process.env.POSTMARK_SENDER_EMAIL ?? "amy@asksymple.ai",
            To: state.organizer_email,
            Subject: subject,
            TextBody: fullEmailBody,
            ReplyTo: state._webhook_target_address,
            MessageStream: "outbound" 
        });
        console.log(`Email sent successfully. MessageID: ${sendResponse.MessageID}. Reply-To set to: ${state._webhook_target_address}`);

        // Update state
        const sentMessage = new AIMessage({ content: fullEmailBody }); 
        return {
            current_step: "awaiting_organizer_response",
            email_history: [sentMessage],
            last_agent_message_id: sendResponse.MessageID // Return the ID
        };

    } catch (error) {
        console.error("Error drafting or sending availability request:", error);
        return {
            current_step: "error",
            error_message: "Failed to ask organizer for availability."
        };
    }
}

/**
 * Processes the organizer's reply email to extract availability.
 */
async function process_organizer_response(state: AgentState): Promise<Partial<AgentState>> {
    console.log("--- Agent Node: process_organizer_response ---");

    // We need the latest email from the organizer
    // Assuming the webhook handler passes the latest email body
    // into a specific state field, e.g., `latest_email_body`
    // For now, let's simulate getting it from history (this needs refinement)
    const lastMessage = state.email_history[state.email_history.length - 1];
    if (!lastMessage || !(lastMessage instanceof HumanMessage)) {
         console.error("Could not find organizer's reply in history.");
         return { current_step: "error", error_message: "Missing organizer reply." };
    }
    const organizerReplyBody = lastMessage.content as string;

    const parser = new JsonOutputParser<{ availability: string[] }>();

    const prompt = PromptTemplate.fromTemplate(
`You are Amy, an expert meeting scheduling assistant.
The meeting organizer has replied with their availability for the meeting about "{topic}".
Analyze their email reply below and extract the specific dates, times, or time ranges they proposed.
List them as an array of strings.

**CRITICAL**: Respond ONLY with a single, valid JSON object containing ONLY the field "availability" which is an array of strings. EXACTLY as shown in the format instructions. Do not add any other fields or commentary.

Format Instructions:
{format_instructions}

Organizer's Reply Email Body:
{reply_body}

JSON Output:`
    );

    const chain = prompt.pipe(llm).pipe(parser);

    console.log("Analyzing organizer's availability reply...");
    try {
        const analysis = await chain.invoke({
            topic: state.meeting_topic ?? "our meeting",
            reply_body: organizerReplyBody,
            format_instructions: parser.getFormatInstructions(),
        });

        console.log("LLM Availability Analysis Result:", analysis);

        // Decide next step
        const next_step = state.participants && state.participants.length > 0
            ? "get_participant_availability" 
            : "confirm_time"; // If no participants, try to confirm

        return {
            organizer_availability: analysis.availability,
            current_step: next_step
        };

    } catch (error) {
        console.error("Error analyzing organizer availability:", error);
        return {
            current_step: "error",
            error_message: "Failed to parse organizer availability."
        };
    }
}

/**
 * Sends emails to participants asking for their availability.
 */
async function ask_participant_availability(
    state: AgentState, 
    config?: RunnableConfig // Add config parameter
): Promise<Partial<AgentState>> {
    console.log("--- Agent Node: ask_participant_availability ---");
    // Log the full state received by this node
    console.log("Full state received:", JSON.stringify(state, null, 2));

    const { participants, organizer_email, organizer_availability, meeting_topic, _webhook_target_address } = state;
    const sessionId = config?.configurable?.thread_id; // Get sessionId from config

    // Log values after destructuring
    console.log(`Participants: ${JSON.stringify(participants)}`);
    console.log(`Session ID (from config): ${sessionId}`);
    console.log(`Webhook Target: ${_webhook_target_address}`);

    // Perform checks and log results
    if (!sessionId) {
        console.error("Check Failed: Session ID missing from config.");
        return { current_step: "error", error_message: "Session ID missing in config." };
    }
    console.log("Check Passed: Session ID exists.");

    if (!_webhook_target_address) {
         console.error("Check Failed: Webhook target address missing.");
        return { current_step: "error", error_message: "Webhook target address missing for Reply-To." };
    }
    console.log("Check Passed: Webhook target address exists.");

     if (!participants || participants.length === 0) {
        console.log("Check Result: No participants found, skipping participant availability check.");
        return { current_step: "confirm_time" }; 
    }
    console.log("Check Passed: Participants exist.");

    // Re-add LLM setup here
    const draftPrompt = PromptTemplate.fromTemplate(
`You are Amy, an expert meeting scheduling assistant.
Draft a polite email body to a meeting participant.
The organizer ({organizer_email}) proposed a meeting about "{topic}".
Their suggested times/availability are: {organizer_availability}
Ask the participant if any of these times work or for their general availability.
Keep it concise. Do not include greetings or closings.

Email Body Draft:`
    );
    const draftChain = draftPrompt.pipe(llm);

    // Try block starts here
    try {
        console.log("STEP 1: Drafting email body...");
        const availabilityText = organizer_availability?.join(", ") || "not specified yet";
        const emailBodyDraftResult = await draftChain.invoke({
            organizer_email: organizer_email ?? "the organizer",
            topic: meeting_topic ?? "our meeting",
            organizer_availability: availabilityText
        });
        const emailBodyDraft = emailBodyDraftResult.content as string;
        console.log(`STEP 2: Draft complete. Body: ${emailBodyDraft}`);

        const subject = `Meeting Request: ${meeting_topic ?? 'Meeting'} - Availability Check`;
        const agentFromEmail = process.env.POSTMARK_SENDER_EMAIL ?? "amy@asksymple.ai";

        console.log(`STEP 3: Starting loop for participants: ${participants?.join(', ')}`);
        let messagesSentCount = 0;
        for (const participantEmail of participants!) { 
            if (participantEmail.toLowerCase() === organizer_email?.toLowerCase()) {
                console.log(`Skipping participant (is organizer): ${participantEmail}`);
                continue;
            }

             console.log(`STEP 4: Processing participant: ${participantEmail}`);
             const fullEmailBody = `Hi there, \n\n${emailBodyDraft}\n\nBest regards,\nAmy\naskSymple.ai Scheduling Assistant`;

             console.log(`STEP 5: Sending email via Postmark to ${participantEmail}...`);
             const sendResponse = await postmarkClient.sendEmail({
                 From: agentFromEmail,
                 To: participantEmail,
                 Subject: subject,
                 TextBody: fullEmailBody,
                 ReplyTo: _webhook_target_address,
                 MessageStream: "outbound"
             });
             console.log(`STEP 6: Postmark send complete for ${participantEmail}. MessageID: ${sendResponse.MessageID}`);

             console.log(`STEP 7: Saving sent message details to DB for ${participantEmail}...`);
             await saveMessage(sessionId, {
                postmark_message_id: sendResponse.MessageID,
                sender_email: agentFromEmail,
                recipient_email: participantEmail,
                subject: subject,
                body_text: fullEmailBody,
                in_reply_to_message_id: state.last_agent_message_id, // Was this message prompted by the last agent msg?
                message_type: 'ai_agent'
            });
            console.log(`STEP 8: DB save complete for ${participantEmail}.`);
            messagesSentCount++;
        }
        console.log(`STEP 9: Finished loop. Sent ${messagesSentCount} emails.`);

        // Update state
        return {
            current_step: "awaiting_participant_response",
            last_agent_message_id: null 
        };

    } catch (error) {
        console.error("!!! ERROR inside ask_participant_availability try block:", error);
        return {
            current_step: "error",
            error_message: `Failed in ask_participant_availability: ${error instanceof Error ? error.message : String(error)}`
        };
    }
}

/**
 * Processes a participant's reply email to extract availability.
 */
async function process_participant_response(state: AgentState, config?: RunnableConfig): Promise<Partial<AgentState>> {
    console.log("--- Agent Node: process_participant_response ---");
    
    const sessionId = config?.configurable?.thread_id;
    if (!sessionId) {
        return { current_step: "error", error_message: "Session ID missing in config." };
    }

    // Identify the participant who replied
    // Get the latest HUMAN message (should be the participant's reply)
    const lastMessage = state.email_history[state.email_history.length - 1];
    if (!lastMessage || !(lastMessage instanceof HumanMessage)) {
         console.error("Could not find participant's reply in history.");
         return { current_step: "error", error_message: "Missing participant reply message." };
    }
    // We need the actual sender email from the webhook data, which isn't easily accessible here.
    // Workaround: Assume the state contains the necessary info, or enhance state loading.
    // Let's assume for now the webhook handler updated the state correctly before calling this.
    // We'll need to get the sender's email to store their availability correctly.
    // PASSING SENDER FOR NOW - NEEDS REFINEMENT IN index.ts
    const participantEmail = state.initial_sender_email; // !!! THIS IS A HACK - Needs sender from latest email
    const participantReplyBody = lastMessage.content as string;

    if (!participantEmail) {
        return { current_step: "error", error_message: "Could not determine participant email." };
    }
    
    console.log(`Processing reply from participant: ${participantEmail}`);

    // Use LLM to parse availability
    const parser = new JsonOutputParser<{ availability: string[], accepted_proposed: boolean | null }>();
    const prompt = PromptTemplate.fromTemplate(
`You are Amy, an expert meeting scheduling assistant.
A participant ({participant_email}) has replied regarding the meeting "{topic}".
Organizer proposed: {organizer_availability}
Analyze their email reply below. Determine:
1. Did they accept one of the proposed times? (true/false/null)
2. What specific dates/times/availability did they state?

**CRITICAL**: Respond ONLY with a valid JSON object containing fields "availability" (array of strings) and "accepted_proposed" (boolean or null). EXACTLY as shown in the format instructions.

Format Instructions:
{format_instructions}

Participant's Reply Email Body:
{reply_body}

JSON Output:`
    );
    const chain = prompt.pipe(llm).pipe(parser);

    try {
        console.log("Analyzing participant's availability reply...");
        const analysis = await chain.invoke({
            participant_email: participantEmail,
            topic: state.meeting_topic ?? "our meeting",
            organizer_availability: state.organizer_availability?.join(", ") ?? "not specified",
            reply_body: participantReplyBody,
            format_instructions: parser.getFormatInstructions(),
        });
        console.log("LLM Participant Availability Analysis:", analysis);

        // Update participant availability state
        const currentParticipantAvailability = state.participant_availability ?? {};
        currentParticipantAvailability[participantEmail] = analysis.availability;

        // Check if all participants have responded
        const allParticipants = state.participants ?? [];
        // Exclude organizer if they are in participants list by mistake
        const targetParticipants = allParticipants.filter(p => p.toLowerCase() !== state.organizer_email?.toLowerCase()); 
        const respondedParticipants = Object.keys(currentParticipantAvailability);
        const allResponded = targetParticipants.every(p => respondedParticipants.includes(p));

        console.log(`Responded: ${respondedParticipants.length} / ${targetParticipants.length}`);

        let next_step: string;
        if (allResponded) {
            console.log("All participants have responded.");
            // TODO: Add logic to check for conflicts/agreement
            next_step = "confirm_time"; // Placeholder
        } else {
            console.log("Still waiting for other participants.");
            next_step = "awaiting_participant_response"; // Stay in waiting state
        }

        return {
            participant_availability: currentParticipantAvailability,
            current_step: next_step
        };

    } catch (error) {
        console.error("Error analyzing participant availability:", error);
        return {
            current_step: "error",
            error_message: `Failed to parse participant ${participantEmail} availability.`
        };
    }
}

/**
 * Simple router node. It doesn't perform actions, just exists as a state in the graph.
 * The actual routing logic happens in the conditional edge originating FROM this node.
 */
async function route_logic(state: AgentState): Promise<Partial<AgentState>> {
    console.log(`--- Agent Node: route_logic. Current step: ${state.current_step} ---`);
    // This node itself doesn't modify the state.
    // It just needs to return a valid Partial<AgentState>.
    return {}; 
}

/**
 * Explicit final node. Does nothing.
 */
async function graph_end(state: AgentState): Promise<Partial<AgentState>> {
    console.log("--- Agent Node: graph_end --- Reached final state for this run.");
    return {};
}

// --- Graph Construction ---

// Add the nodes to the graph
workflow.addNode("handle_start", handle_start);
workflow.addNode("ask_organizer_availability", ask_organizer_availability);
workflow.addNode("process_organizer_response", process_organizer_response);
workflow.addNode("ask_participant_availability", ask_participant_availability);
workflow.addNode("process_participant_response", process_participant_response);
workflow.addNode("route_logic", route_logic);
workflow.addNode("graph_end", graph_end); // Add the end node

// Set the entry point
// @ts-ignore
workflow.setEntryPoint("handle_start");

// Always go to the router after the entry point node
// @ts-ignore
workflow.addEdge("handle_start", "route_logic");

// The router node conditionally decides where to go next
// @ts-ignore
workflow.addConditionalEdges(
    "route_logic" as any, // Source node is the router
    async (state: AgentState) => {
        // Decision function returns the NAME of the next node
        console.log(`--- Agent Edge: Deciding from route_logic. Current step: ${state.current_step} ---`);
        switch (state.current_step) {
            case "get_organizer_availability":
                return "ask_organizer_availability";
            case "process_organizer_response": 
                 return "process_organizer_response"; 
            case "get_participant_availability":
                 return "ask_participant_availability";
            case "process_participant_response":
                 return "process_participant_response";
            case "awaiting_organizer_response": 
            case "awaiting_participant_response":
            case "error":
            case "end_other_intent":
            default:
                 console.log(`Routing to graph_end from router due to step: ${state.current_step}`);
                 return "graph_end"; // Route to explicit end node
        }
    },
    {
        // Mapping the NAMES returned above to the actual nodes
        "ask_organizer_availability": "ask_organizer_availability" as any,
        "process_organizer_response": "process_organizer_response" as any,
        "ask_participant_availability": "ask_participant_availability" as any,
        "process_participant_response": "process_participant_response" as any,
        "graph_end": "graph_end" as any // Map to the explicit end node
    }
);

// After action nodes, always return to the router to decide the next step
// @ts-ignore
workflow.addEdge("ask_organizer_availability", "route_logic");
// @ts-ignore
workflow.addEdge("process_organizer_response", "route_logic");
// @ts-ignore
workflow.addEdge("ask_participant_availability", "route_logic");
// @ts-ignore
workflow.addEdge("process_participant_response", "route_logic");
// TODO: Add edges from future action nodes back to route_logic

// Explicit edge from graph_end to the final END state
// @ts-ignore
workflow.addEdge("graph_end", END);


// Compile the graph into a runnable App
export const app = workflow.compile();

console.log("LangGraph workflow compiled.");

// We will add nodes and the graph definition below this later 

// Define the expected JSON structure from the LLM
interface InitialAnalysis {
    intent: "schedule_meeting" | "other";
    topic: string | null;
    participants: string[] | null; // List of email addresses
    proposed_times: string[] | null; // Extracted time suggestions
}

// Initialize the OpenAI model
// Make sure OPENAI_API_KEY is set in your .env file
const llm = new ChatOpenAI({ model: "gpt-4o-mini", temperature: 0 });

// Initialize Postmark Client
// Make sure POSTMARK_SERVER_TOKEN is set in your .env file
const postmarkToken = process.env.POSTMARK_SERVER_TOKEN ?? "";
if (!postmarkToken) {
    console.warn("POSTMARK_SERVER_TOKEN is not set. Email sending will fail.");
}
const postmarkClient = new PostmarkClient(postmarkToken);
const senderEmailAddress = process.env.POSTMARK_SENDER_EMAIL ?? "amy@asksymple.ai"; // The 'From' address 