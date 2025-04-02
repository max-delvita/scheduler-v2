import { createClient } from '@supabase/supabase-js';
import 'dotenv/config';

// Initialize Supabase Client
const supabaseUrl = process.env.SUPABASE_URL;
const supabaseAnonKey = process.env.SUPABASE_ANON_KEY;

if (!supabaseUrl || !supabaseAnonKey) {
  console.error("Supabase URL or Anon Key not found in .env file. Database operations will fail.");
  throw new Error("Missing Supabase credentials."); // Throw error to prevent startup
}

export const supabase = createClient(supabaseUrl!, supabaseAnonKey!); 
console.log("Supabase client initialized (db.ts).");


/**
 * Saves a message record to the session_messages table.
 */
export async function saveMessage(sessionId: string, message: Record<string, any>) {
    if (!sessionId) {
        console.error("SaveMessage Error: Cannot save message without session ID.");
        return;
    }
    const messageIdToSave = message.postmark_message_id;
    console.log(`Attempting to save message [${messageIdToSave}] for session ${sessionId}, type: ${message.message_type}`);
    try {
        // Explicitly select columns to ensure we don't pass extra fields
        const messageData = {
            session_id: sessionId,
            postmark_message_id: message.postmark_message_id,
            sender_email: message.sender_email,
            recipient_email: message.recipient_email,
            subject: message.subject,
            body_text: message.body_text,
            in_reply_to_message_id: message.in_reply_to_message_id,
            message_type: message.message_type,
            // received_at has default value in DB
        };

        const { data, error } = await supabase
            .from('session_messages')
            .insert(messageData) // Pass the structured data
            .select(); // Select the inserted row to confirm
            
        if (error) {
            console.error(`DB Error saving message [${messageIdToSave}] for session ${sessionId}:`, error);
        } else {
            console.log(`Successfully saved message [${messageIdToSave}] for session ${sessionId}. DB returned:`, data); // Log returned data
        }
    } catch (err) {
        console.error(`Exception saving message [${messageIdToSave}] for session ${sessionId}:`, err);
    }
} 