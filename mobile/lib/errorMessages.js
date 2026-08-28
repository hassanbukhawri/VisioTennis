// Phase C (2026-08-01): human-readable titles for each of the backend's
// structured error_code values (see backend/app.py's HTTPException detail
// shapes). The backend's own `message` text is already written to be shown
// to a user as-is and carries the specific, actionable guidance -- this
// module only supplies the short category title shown above it, so each of
// the 5 hardened failure modes reads as a distinct screen state rather than
// one undifferentiated error box.
export const UPLOAD_ERROR_TITLES = {
  bad_video_file: "There's a problem with this video file",
  no_player_detected: "No player detected in this video",
  low_pose_detection: "Couldn't get a clear read on the player's movement",
  not_tennis: "This doesn't look like tennis footage",
};

export const UNKNOWN_ERROR_TITLE = "Something went wrong";

export const COACHING_UNAVAILABLE_MESSAGE =
  "Coaching feedback isn't available right now. Your stroke analysis above is still valid -- " +
  "this is a temporary issue on our end, not a problem with your video. Try again in a few minutes.";
