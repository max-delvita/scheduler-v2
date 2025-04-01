import { BaseMessage } from '@langchain/core/messages';
import { StateGraph, END } from "@langchain/langgraph";
import { ChatOpenAI } from "@langchain/openai";
import { JsonOutputParser } from "@langchain/core/output_parsers";
import { PromptTemplate } from "@langchain/core/prompts";
import { Client as PostmarkClient } from "postmark"; // Import Postmark client
import { AIMessage, HumanMessage } from "@langchain/core/messages";

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
Analyze the following email content (Subject and Body) from a potential organizer ({sender}).
Your goal is to understand:
1.  **Intent**: Is the primary intent to schedule a new meeting? Use "schedule_meeting" or "other".
2.  **Topic**: What is the meeting about? (e.g., "Project Alpha Sync")
3.  **Participants**: List any email addresses explicitly mentioned as participants.

**CRITICAL**: Respond ONLY with a single, valid JSON object containing ONLY the fields "intent", "topic", and "participants" EXACTLY as shown in the format instructions. Do NOT add any other fields or nesting.

Format Instructions:
{format_instructions}

Email Details:
Subject: {subject}
Sender: {sender}
Body:
{body}

JSON Output:`
    );

    // Create a chain: Prompt -> LLM -> Parser
    const chain = prompt.pipe(llm).pipe(parser);

    console.log("Analyzing initial email with LLM...");
    try {
        const analysis: InitialAnalysis = await chain.invoke({
            sender: state.initial_sender_email,
            subject: state.initial_subject ?? "", // Use empty string if null
            body: state.initial_body ?? "", // Use empty string if null
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

        // Update state based on analysis
        return {
            organizer_email: state.initial_sender_email,
            meeting_topic: analysis.topic,
            participants: analysis.participants,
            current_step: "get_organizer_availability" // Ensure this step triggers next action via router
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

        // Send email via Postmark
        console.log(`Sending availability request to ${state.organizer_email}...`);
        const sendResponse = await postmarkClient.sendEmail({
            From: senderEmailAddress,
            To: state.organizer_email,
            Subject: subject,
            TextBody: fullEmailBody, 
            MessageStream: "outbound" 
        });
        console.log("Email sent successfully. MessageID:", sendResponse.MessageID);

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
 * Simple router node. It doesn't perform actions, just exists as a state in the graph.
 * The actual routing logic happens in the conditional edge originating FROM this node.
 */
async function route_logic(state: AgentState): Promise<Partial<AgentState>> {
    console.log(`--- Agent Node: route_logic. Current step: ${state.current_step} ---`);
    // This node itself doesn't modify the state.
    // It just needs to return a valid Partial<AgentState>.
    return {}; 
}

// --- Graph Construction ---

// Add the nodes to the graph
workflow.addNode("handle_start", handle_start);
workflow.addNode("ask_organizer_availability", ask_organizer_availability);
workflow.addNode("process_organizer_response", process_organizer_response);
workflow.addNode("route_logic", route_logic);

// Set the entry point
// @ts-ignore
workflow.setEntryPoint("handle_start");

// Always go to the router after the entry point node
// @ts-ignore
workflow.addEdge("handle_start", "route_logic");

// The router node conditionally decides where to go next
// @ts-ignore
workflow.addConditionalEdges(
    "route_logic" as any, // Source node is the router - CAST TO ANY
    async (state: AgentState) => {
        // Decision function returns the NAME of the next node or END
        console.log(`--- Agent Edge: Deciding from route_logic. Current step: ${state.current_step} ---`);
        switch (state.current_step) {
            case "get_organizer_availability":
                return "ask_organizer_availability";
            case "process_organizer_response": 
                 return "process_organizer_response"; 
            // --- Add cases for future steps here ---
            // case "get_participant_availability":
            //     return "ask_participant_availability";
            // case "confirm_time":
            //     return "confirm_time";
            // case "send_confirmation_email":
            //     return "send_confirmation_email";
            // ---------------------------------------
            case "awaiting_organizer_response": 
            case "awaiting_participant_response":
            case "error":
            case "end_other_intent":
            default:
                 console.log(`Routing to END from router due to step: ${state.current_step}`);
                 return END;
        }
    },
    {
        // Mapping the NAMES returned above to the actual nodes
        "ask_organizer_availability": "ask_organizer_availability" as any,
        "process_organizer_response": "process_organizer_response" as any,
        // TODO: Add mappings for future nodes here
        END: END 
    }
);

// After action nodes, always return to the router to decide the next step
// @ts-ignore
workflow.addEdge("ask_organizer_availability", "route_logic");
// @ts-ignore
workflow.addEdge("process_organizer_response", "route_logic");
// TODO: Add edges from future action nodes back to route_logic


// Compile the graph into a runnable App
export const app = workflow.compile();

console.log("LangGraph workflow compiled.");

// We will add nodes and the graph definition below this later 

// Define the expected JSON structure from the LLM
interface InitialAnalysis {
    intent: "schedule_meeting" | "other";
    topic: string | null;
    participants: string[] | null; // List of email addresses
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