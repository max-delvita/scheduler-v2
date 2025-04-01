import { Elysia } from 'elysia';
import 'dotenv/config'; // Load environment variables from .env file
import { app as agentApp } from './src/agent'; // Import the compiled LangGraph app
import { HumanMessage, AIMessage, BaseMessage } from '@langchain/core/messages'; // Import message types
import { createClient } from '@supabase/supabase-js'; // Import Supabase client
import type { AgentState } from './src/agent'; // Use type import

// Initialize Supabase Client
const supabaseUrl = process.env.SUPABASE_URL;
const supabaseAnonKey = process.env.SUPABASE_ANON_KEY;

if (!supabaseUrl || !supabaseAnonKey) {
  console.error("Supabase URL or Anon Key not found in .env file. Database operations will fail.");
  // Optionally exit or handle this case more gracefully
}

const supabase = createClient(supabaseUrl!, supabaseAnonKey!); // Use non-null assertion as we checked above
console.log("Supabase client initialized.");

// --- Database Helper Functions ---

async function findSessionByReply(messageId: string | null): Promise<string | null> {
  if (!messageId) return null;
  try {
    const { data, error } = await supabase
      .from('session_messages')
      .select('session_id')
      .eq('postmark_message_id', messageId)
      // .eq('message_type', 'ai_agent') // Optional: Ensure it replies to an agent msg
      .limit(1)
      .single();
    
    if (error && error.code !== 'PGRST116') { // Ignore "Row not found" errors
      console.error("Error finding session by reply:", error);
    }
    return data?.session_id ?? null;
  } catch (err) {
    console.error("Exception finding session by reply:", err);
    return null;
  }
}

async function getSessionState(sessionId: string): Promise<AgentState | null> {
  try {
    const { data: sessionData, error: sessionError } = await supabase
      .from('scheduling_sessions')
      .select('*')
      .eq('session_id', sessionId)
      .single();

    if (sessionError) {
      console.error("Error fetching session:", sessionError);
      return null;
    }
    if (!sessionData) return null;

    const { data: messagesData, error: messagesError } = await supabase
      .from('session_messages')
      .select('message_type, body_text')
      .eq('session_id', sessionId)
      .order('received_at', { ascending: true });

    if (messagesError) {
      console.error("Error fetching messages:", messagesError);
      return null; // Or return partial state?
    }

    // Reconstruct email history
    const email_history: BaseMessage[] = (messagesData ?? []).map(msg => {
      if (msg.message_type?.startsWith('human')) {
        return new HumanMessage({ content: msg.body_text ?? "" });
      } else if (msg.message_type === 'ai_agent') {
        return new AIMessage({ content: msg.body_text ?? "" });
      }
      return new HumanMessage({ content: msg.body_text ?? "" }); // Default fallback
    });

    // Map DB columns to AgentState interface
    const agentState: AgentState = {
      initial_sender_email: sessionData.organizer_email, // Assuming organizer starts
      initial_sender_name: null, // Not currently stored, maybe add?
      initial_subject: null, // Not specific to state, but could be stored?
      initial_body: null, // Handled by history
      organizer_email: sessionData.organizer_email,
      organizer_availability: sessionData.organizer_availability_raw?.split('\n') ?? null, // Simple split example
      participants: sessionData.participants,
      participant_availability: sessionData.participant_availability_json as Record<string, string[]> ?? null,
      confirmed_datetime: sessionData.confirmed_datetime_utc ? new Date(sessionData.confirmed_datetime_utc) : null,
      meeting_topic: sessionData.meeting_topic,
      current_step: sessionData.current_step,
      error_message: null, // Clear error on load?
      email_history: email_history,
    };

    return agentState;

  } catch (err) {
    console.error("Exception fetching session state:", err);
    return null;
  }
}

async function saveSessionState(sessionId: string, state: Partial<AgentState>) {
  try {
    const updates: Record<string, any> = {
        current_step: state.current_step,
        status: state.current_step === 'error' ? 'error' : (state.current_step === 'send_confirmation_email' ? 'confirmed' : 'pending'), // Example status update
        organizer_availability_raw: state.organizer_availability?.join('\n'),
        participant_availability_json: state.participant_availability,
        confirmed_datetime_utc: state.confirmed_datetime?.toISOString(),
        meeting_topic: state.meeting_topic,
        participants: state.participants,
        // updated_at is handled by trigger
    };

    // Remove undefined keys to avoid overwriting DB columns with null
    Object.keys(updates).forEach(key => updates[key] === undefined && delete updates[key]);

    const { error } = await supabase
      .from('scheduling_sessions')
      .update(updates)
      .eq('session_id', sessionId);

    if (error) {
        console.error("Error updating session state:", error);
    }
  } catch (err) {
    console.error("Exception saving session state:", err);
  }
}

async function saveMessage(sessionId: string, message: Record<string, any>) {
    try {
        const { error } = await supabase
            .from('session_messages')
            .insert({ session_id: sessionId, ...message });
        if (error) {
            console.error("Error saving message:", error);
        }
    } catch (err) {
        console.error("Exception saving message:", err);
    }
}

// --- Elysia Server Setup ---

const app = new Elysia();

// Define the webhook endpoint for Postmark
app.post('/webhook/email', async ({ body, request }) => { // Access original request for headers
  console.log('--- Received Email Webhook ---');

  const postmarkBody = body as any;

  // Extract more details, including Message-ID and In-Reply-To
  const senderEmail = postmarkBody.FromFull?.Email;
  const senderName = postmarkBody.FromFull?.Name;
  const subject = postmarkBody.Subject;
  const textBody = postmarkBody.TextBody;
  const recipient = postmarkBody.OriginalRecipient;
  const messageId = postmarkBody.MessageID;
  // Find In-Reply-To header (case-insensitive)
  const headers = postmarkBody.Headers as {Name: string, Value: string}[];
  const inReplyToHeader = headers?.find(h => h.Name.toLowerCase() === 'in-reply-to');
  const inReplyToId = inReplyToHeader?.Value?.replace(/[<>]/g, ''); // Clean potential <> wrapping

  console.log(`Recipient: ${recipient}`);
  console.log(`Sender: ${senderName} <${senderEmail}>`);
  console.log(`Subject: ${subject}`);
  console.log(`Message-ID: ${messageId}`);
  console.log(`In-Reply-To: ${inReplyToId}`);
  console.log(`Body (Text):\n${textBody?.substring(0, 100)}...`); // Log snippet

  // Basic validation
  if (!senderEmail || !textBody || !messageId) {
    console.error("Missing sender, body, or messageId. Cannot process.");
    return new Response('Missing data', { status: 400 });
  }

  let sessionId: string | null = null;
  let loadedState: AgentState | null = null;

  // --- Check if it's a reply --- 
  sessionId = await findSessionByReply(inReplyToId ?? null);
  
  if (sessionId) {
      console.log(`Identified as reply to session: ${sessionId}`);
      loadedState = await getSessionState(sessionId);
      if (loadedState) {
          console.log(`Loaded state for session ${sessionId}, current step: ${loadedState.current_step}`);
          // Update step to indicate we should process the reply NOW
          switch(loadedState.current_step) {
              case "awaiting_organizer_response":
                  loadedState.current_step = "process_organizer_response"; 
                  break;
              case "awaiting_participant_response": // Placeholder for later
                  // loadedState.current_step = "process_participant_responses";
                  console.log("Participant response handling not implemented yet.")
                  loadedState = null; // Mark as ignore by nulling loadedState
                  break;
              default:
                  console.warn(`Reply received for session ${sessionId} in unexpected state: ${loadedState.current_step}. Ignoring reply.`);
                  loadedState = null; // Mark as ignore
          }
      } else {
          console.error(`Could not load state for session ${sessionId} found via reply-to ${inReplyToId}. Ignoring.`);
          sessionId = null; // Reset session ID as state load failed
      }
  } else {
      console.log("Email is not identified as a reply. Treating as new request.");
  }

  // --- Agent Invocation --- 
  if (loadedState || !sessionId) { // Proceed if loaded state exists OR it's a new session
    console.log(`--- Invoking Agent ---`); // Simplified log
    try {
      let initialState: Partial<AgentState>;
      let currentSessionId = sessionId; 

      if (loadedState && currentSessionId) {
          // Resuming existing session
          console.log(`Resuming session ${currentSessionId} at step: ${loadedState.current_step}`);
          initialState = {
              ...loadedState,
              // Overwrite history with full loaded history PLUS new message
              email_history: loadedState.email_history.concat([new HumanMessage({ content: textBody })]), 
          };
          // No need to determine startNode, graph handles it via routing
      } else {
          // Starting new session
          console.log("Starting new session.");
          initialState = {
              initial_sender_email: senderEmail,
              initial_sender_name: senderName,
              initial_subject: subject,
              initial_body: textBody,
              email_history: [new HumanMessage({ content: textBody })],
              current_step: 'start'
          };
          
          // Create new session in DB *before* invoking agent
          const { data: newSession, error: createError } = await supabase
              .from('scheduling_sessions')
              .insert({ organizer_email: senderEmail, current_step: 'start' })
              .select('session_id')
              .single();
              
          if (createError || !newSession) {
              console.error("Failed to create new session in DB:", createError);
              throw new Error("Failed to create DB session.");
          }
          currentSessionId = newSession.session_id;
          console.log(`Created new session: ${currentSessionId}`);
      }
      
      // Save incoming message before invoking agent
      await saveMessage(currentSessionId!, {
          postmark_message_id: messageId,
          sender_email: senderEmail,
          recipient_email: recipient,
          subject: subject,
          body_text: textBody,
          in_reply_to_message_id: inReplyToId,
          message_type: loadedState ? `human_${senderEmail === loadedState.organizer_email ? 'organizer' : 'participant'}` : 'human_organizer' // Determine type
      });

      // Invoke the agent workflow
      const finalState = await agentApp.invoke(initialState, { 
          configurable: { thread_id: currentSessionId! } // Non-null assertion okay here
      });

      console.log('--- Agent Finished ---');
      console.log('Final Agent State Keys:', Object.keys(finalState)); // Log keys to see what agent returned

      // --- Save Final State & Messages --- 
      console.log(`Saving final state for session ${currentSessionId}. New step: ${finalState.current_step}`);
      await saveSessionState(currentSessionId!, finalState);
      
      // Save the AI message(s) that were generated during this run
      if (finalState.last_agent_message_id) {
        // Find the AI message in the history added in the *last step*
        // Note: This assumes the last message added by the agent run is the one we want.
        // More robust logic might involve comparing history before/after invoke.
        const agentMessages = finalState.email_history?.filter((msg: BaseMessage) => msg instanceof AIMessage) ?? [];
        const lastAgentMessage = agentMessages.length > 0 ? agentMessages[agentMessages.length - 1] : null;

        if (lastAgentMessage) {
            console.log(`Saving agent message with ID: ${finalState.last_agent_message_id}`);
            await saveMessage(currentSessionId!, {
                postmark_message_id: finalState.last_agent_message_id, // The ID from Postmark
                sender_email: process.env.POSTMARK_SENDER_EMAIL ?? "amy@asksymple.ai", // Get from env directly
                recipient_email: finalState.organizer_email, // Assuming it was sent to organizer
                subject: `Availability for ${finalState.meeting_topic ?? 'meeting'}`, // Reconstruct or get from state?
                body_text: lastAgentMessage.content as string,
                in_reply_to_message_id: messageId, // The AI message is in reply to the triggering human message
                message_type: 'ai_agent'
            });
        } else {
             console.warn("last_agent_message_id was present in state, but couldn't find corresponding AIMessage in history.");
        }
      }

    } catch (error) {
      console.error("Error during agent invocation or state saving:", error);
      // Optionally update session status to 'error' in DB
      if (sessionId) {
          await supabase.from('scheduling_sessions').update({ status: 'error' }).eq('session_id', sessionId);
      }
    }
  } else {
      console.log("Ignoring this email - not invoking agent.");
  }
  // --- End Agent Invocation ---

  console.log('--- Finished Processing Webhook ---');
  return new Response('Webhook received successfully', { status: 200 });
});

// Start the server
const PORT = process.env.PORT || 3000;
app.listen(PORT, () => {
  console.log(
    `🦊 Elysia server running at http://${app.server?.hostname}:${app.server?.port}`
  );
  console.log(`Webhook endpoint available at POST /webhook/email`);
});

console.log('Server setup complete.'); // This line might not be reached if listen is blocking