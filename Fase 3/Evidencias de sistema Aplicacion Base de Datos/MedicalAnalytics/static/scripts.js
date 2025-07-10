// scripts.js
document.addEventListener('DOMContentLoaded', function() {
    console.log("Chat cargado");

    function addMessage(sender, text, type) {
        const chatBox = document.getElementById('chat-box');
        const msgDiv = document.createElement('div');
        msgDiv.className = `fade-in ${type === 'user' ? 'user-message' : 'bot-message'} p-3 mb-2 rounded`;
        
        // Formatear texto médico con mejor presentación
        let formattedText = text
            .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')  // Negritas
            .replace(/\*(.*?)\*/g, '<em>$1</em>')              // Cursivas
            .replace(/---(.*?)---/g, '<hr><h5>$1</h5>')        // Encabezados
            .replace(/\n/g, '<br>');                           // Saltos de línea
        
        msgDiv.innerHTML = `
            <strong>${sender}:</strong>
            <div class="mt-1">${formattedText}</div>
        `;
        chatBox.appendChild(msgDiv);
        chatBox.scrollTop = chatBox.scrollHeight;
        return msgDiv;
    }

    function showLoading() {
        const chatBox = document.getElementById('chat-box');
        const loadingDiv = document.createElement('div');
        loadingDiv.className = 'fade-in bot-message p-3 mb-2 rounded';
        loadingDiv.id = 'loading-indicator';
        loadingDiv.innerHTML = `
            <div class="d-flex align-items-center">
                <div class="spinner-border spinner-border-sm text-primary me-2" role="status">
                    <span class="visually-hidden">Cargando...</span>
                </div>
                <span>Buscando información...</span>
            </div>
        `;
        chatBox.appendChild(loadingDiv);
        chatBox.scrollTop = chatBox.scrollHeight;
        return loadingDiv;
    }

    function hideLoading(loadingElement) {
        if (loadingElement) {
            loadingElement.remove();
        }
    }

    window.sendMessage = async function() {
        const input = document.getElementById('user-input');
        const message = input.value.trim();
        if (!message) return;

        addMessage('Tú', message, 'user');
        input.value = '';
        
        // Mostrar indicador de carga
        const loadingElement = showLoading();
        
        try {
            const response = await fetch('/chat', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({ mensaje: message })
            });
            
            const data = await response.json();
            
            // Eliminar mensaje de carga
            hideLoading(loadingElement);
            
            // Mostrar respuesta formateada
            addMessage('Asistente IA', data.respuesta, 'bot');
        } catch (error) {
            hideLoading(loadingElement);
            addMessage('Sistema', '⚠️ Error de conexión con el asistente', 'bot');
        }
    }

    document.getElementById('user-input').addEventListener('keypress', (e) => {
        if (e.key === 'Enter') {
            e.preventDefault();
            window.sendMessage();
        }
    });
});