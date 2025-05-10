document.addEventListener('DOMContentLoaded', function() {
    console.log("Dashboard cargado");

    function addMessage(sender, text, type) {
        const chatBox = document.getElementById('chat-box');
        const msgDiv = document.createElement('div');
        msgDiv.className = `fade-in ${type === 'user' ? 'user-message' : 'bot-message'} p-3 mb-2 rounded`;
        msgDiv.innerHTML = `
            <strong>${sender}:</strong>
            <div class="mt-1">${text.replace(/\n/g, '<br>')}</div>
        `;
        chatBox.appendChild(msgDiv);
        chatBox.scrollTop = chatBox.scrollHeight;
    }

    window.sendMessage = async function() {
        const input = document.getElementById('user-input');
        const message = input.value.trim();
        if (!message) return;

        addMessage('Tú', message, 'user');
        input.value = '';
        
        try {
            const response = await fetch('/chat', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({ mensaje: message })
            });
            
            const data = await response.json();
            addMessage('Asistente IA', data.respuesta, 'bot');
        } catch (error) {
            addMessage('Sistema', 'Error de conexión con Ollama', 'bot');
        }
    }

    document.getElementById('user-input').addEventListener('keypress', (e) => {
        if (e.key === 'Enter') {
            e.preventDefault();
            window.sendMessage();
        }
    });
});