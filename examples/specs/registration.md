# User registration webhook

Verify that registration through the GUI records a webhook.

1. GUI browser: Open the registration form, enter user ada, and submit.
   Assertion: contains:registered

2. CLI: Confirm the webhook file contains that user.
   Assertion: json:user=ada
