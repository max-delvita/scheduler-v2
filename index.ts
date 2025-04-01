import { Elysia } from 'elysia';
import 'dotenv/config'; // Load environment variables from .env file
import { app as agentApp } from './src/agent'; // Import the compiled LangGraph app
import { HumanMessage } from '@langchain/core/messages'; // Import message type

// Initialize Elysia app
const app = new Elysia();

// Define the webhook endpoint for Postmark
// Postmark sends inbound emails as POST requests
app.post('/webhook/email', async ({ body }) => {
  console.log('--- Received Email Webhook ---');

  // Assuming the body matches the Postmark inbound format
  // It's good practice to add type checking/validation here later
  const postmarkBody = body as any; // Use 'as any' for now, or define an interface

  // Extract relevant information
  const senderEmail = postmarkBody.FromFull?.Email;
  const senderName = postmarkBody.FromFull?.Name;
  const subject = postmarkBody.Subject;
  const textBody = postmarkBody.TextBody;
  const recipient = postmarkBody.OriginalRecipient; // Or check ToFull array

  console.log(`Recipient: ${recipient}`);
  console.log(`Sender: ${senderName} <${senderEmail}>`);
  console.log(`Subject: ${subject}`);
  console.log(`Body (Text):\n${textBody}`);

  // Basic validation
  if (!senderEmail || !textBody) {
    console.error("Missing sender email or body, cannot process.");
    return new Response('Missing data', { status: 400 }); // Bad Request
  }

  // --- Trigger LangGraph Agent --- 
  console.log('--- Invoking Agent ---');
  try {
    // Prepare the initial state for the agent
    const initialState = {
        initial_sender_email: senderEmail,
        initial_sender_name: senderName,
        initial_subject: subject,
        initial_body: textBody,
        // Add the first email to the history
        email_history: [new HumanMessage({ content: textBody })]
    };

    // Invoke the agent workflow
    // The .invoke() method runs the graph from the entry point until it hits END
    const finalState = await agentApp.invoke(initialState);

    console.log('--- Agent Finished ---');
    console.log('Final Agent State:', finalState);
    // TODO: Based on finalState.current_step or other fields,
    // potentially send an email reply using Postmark

  } catch (error) {
    console.error("Error invoking agent:", error);
    // Decide if you want to return a 500 error to Postmark
    // return new Response('Agent processing error', { status: 500 });
  }
  // --- End Agent Trigger ---

  console.log('--- Finished Processing Webhook ---');

  // Respond to Postmark to acknowledge receipt
  // Postmark expects a 200 OK response
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