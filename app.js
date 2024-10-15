// app.js

// This function can be used to add any interactive functionality you want later.
// For now, it just logs a message when the document is ready.
document.addEventListener('DOMContentLoaded', function() {
    console.log('Document loaded and ready.');
});

// Optional: Add form validation or other interactivity if needed
const form = document.querySelector('form');

form.addEventListener('submit', function(event) {
    // Example: Alert the user when the form is submitted
    alert('Form submitted! Processing your data...');
});
