document.addEventListener('DOMContentLoaded', () => {
    const uploadButton = document.getElementById('uploadButton');
    const imageInput = document.getElementById('imageInput');
    const preview = document.getElementById('preview');
    const results = document.getElementById('results');

    uploadButton.addEventListener('click', () => {
        imageInput.click();
    });

    imageInput.addEventListener('change', async (e) => {
        const file = e.target.files[0];
        if (!file) return;

        // Mostrar preview
        preview.innerHTML = `<img src="${URL.createObjectURL(file)}" alt="Preview">`;

        // Preparar formulario
        const formData = new FormData();
        formData.append('file', file);

        try {
            // Mostrar loading
            results.innerHTML = '<p>Procesando imagen...</p>';

            // Enviar imagen
            const response = await fetch('/predict', {
                method: 'POST',
                body: formData
            });

            const prediction = await response.json();

            // Mostrar resultados
            results.innerHTML = `
                <h2>Resultados:</h2>
                <p>Clase: ${prediction.class}</p>
                <p>Confianza: ${(prediction.confidence * 100).toFixed(2)}%</p>
                <h3>Probabilidades:</h3>
                ${Object.entries(prediction.probabilities)
                    .map(([className, prob]) => `
                        <p>${className}: 
                            <div class="confidence-bar">
                                <div class="confidence-fill" style="width: ${prob * 100}%"></div>
                            </div>
                            ${(prob * 100).toFixed(2)}%
                        </p>
                    `).join('')}
            `;
        } catch (error) {
            results.innerHTML = `<p class="error">Error: ${error.message}</p>`;
        }
    });
});