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
        console.error("Cannot save message without session ID.");
        return;
    }
    console.log(`Attempting to save message for session ${sessionId}, type: ${message.message_type}`);
    try {
        const { error } = await supabase
            .from('session_messages')
            .insert({ session_id: sessionId, ...message });
        if (error) {
            console.error("Error saving message to DB:", error);
        } else {
            console.log(`Successfully saved message ID ${message.postmark_message_id} for session ${sessionId}.`)
        }
    } catch (err) {
        console.error("Exception saving message:", err);
    }
} 