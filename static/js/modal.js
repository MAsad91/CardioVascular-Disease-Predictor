// Modal System for Messages
class ModalSystem {
    constructor() {
        this.modalContainer = null;
        this.modalQueue = [];
        this.isModalOpen = false;
        this.init();
    }

    init() {
        // Create modal container if it doesn't exist
        if (!document.getElementById('modal-container')) {
            this.modalContainer = document.createElement('div');
            this.modalContainer.id = 'modal-container';
            this.modalContainer.style.cssText = `
                position: fixed;
                top: 0;
                left: 0;
                width: 100%;
                height: 100%;
                background: rgba(0, 0, 0, 0.5);
                display: none;
                justify-content: center;
                align-items: center;
                z-index: 9999;
                backdrop-filter: blur(5px);
            `;
            document.body.appendChild(this.modalContainer);
        } else {
            this.modalContainer = document.getElementById('modal-container');
        }
    }

    showModal(title, message, type = 'info', options = {}) {
        // Add to queue instead of showing immediately if a modal is already open
        if (this.isModalOpen) {
            this.modalQueue.push({ title, message, type, options });
            return;
        }

        this.isModalOpen = true;
        const modal = document.createElement('div');
        modal.className = 'modal-dialog';
        modal.style.cssText = `
            background: white;
            border-radius: 15px;
            padding: 0;
            max-width: 500px;
            width: 90%;
            box-shadow: 0 20px 60px rgba(0, 0, 0, 0.3);
            transform: scale(0.7);
            opacity: 0;
            transition: all 0.3s ease;
            position: relative;
        `;

        const iconMap = {
            'success': 'bi-check-circle-fill',
            'error': 'bi-exclamation-triangle-fill',
            'warning': 'bi-exclamation-triangle-fill',
            'info': 'bi-info-circle-fill'
        };

        const colorMap = {
            'success': '#10b981',
            'error': '#ef4444',
            'warning': '#f59e0b',
            'info': '#7ecbff'
        };

        const icon = iconMap[type] || iconMap.info;
        const color = colorMap[type] || colorMap.info;

        modal.innerHTML = `
            <div class="modal-header" style="padding: 1.5rem 1.5rem 0 1.5rem; border: none; position: relative;">
                <div class="d-flex align-items-center">
                    <i class="bi ${icon} me-3" style="font-size: 1.5rem; color: ${color};"></i>
                    <h5 class="modal-title mb-0" style="color: #333; font-weight: 600;">${title}</h5>
                </div>
                <button type="button" class="modal-close-btn" style="background: none; border: none; font-size: 1.5rem; color: #666; cursor: pointer; position: absolute; top: 1rem; right: 1rem; width: 30px; height: 30px; display: flex; align-items: center; justify-content: center; border-radius: 50%; transition: background-color 0.2s;">×</button>
            </div>
            <div class="modal-body" style="padding: 1rem 1.5rem 1.5rem 1.5rem;">
                <p style="color: #666; margin: 0; line-height: 1.6;">${message}</p>
            </div>
            ${options.showButtons ? `
            <div class="modal-footer" style="padding: 0 1.5rem 1.5rem 1.5rem; border: none;">
                ${options.cancelText ? `<button type="button" class="btn btn-secondary" onclick="modalSystem.closeModal(this)">${options.cancelText}</button>` : ''}
                <button type="button" class="btn btn-primary" onclick="modalSystem.closeModal(this)">${options.confirmText || 'OK'}</button>
            </div>
            ` : ''}
        `;

        // Clear any existing modals before adding new one
        this.modalContainer.innerHTML = '';
        this.modalContainer.appendChild(modal);
        this.modalContainer.style.display = 'flex';

        // Add click outside to close functionality
        this.modalContainer.addEventListener('click', (e) => {
            if (e.target === this.modalContainer) {
                this.closeModal(modal);
            }
        });

        // Add keyboard event listener for Escape key
        const handleEscape = (e) => {
            if (e.key === 'Escape') {
                this.closeModal(modal);
                document.removeEventListener('keydown', handleEscape);
            }
        };
        document.addEventListener('keydown', handleEscape);

        // Add close button functionality
        const closeBtn = modal.querySelector('.modal-close-btn');
        if (closeBtn) {
            closeBtn.addEventListener('click', () => {
                this.closeModal(modal);
            });
            
            // Add hover effects
            closeBtn.addEventListener('mouseenter', () => {
                closeBtn.style.backgroundColor = 'rgba(0, 0, 0, 0.1)';
            });
            
            closeBtn.addEventListener('mouseleave', () => {
                closeBtn.style.backgroundColor = 'transparent';
            });
        }

        // Animate in
        setTimeout(() => {
            modal.style.transform = 'scale(1)';
            modal.style.opacity = '1';
        }, 10);

        // Auto close after delay if specified
        if (options.autoClose) {
            setTimeout(() => {
                this.closeModal(modal);
            }, options.autoClose);
        }

        return modal;
    }

    closeModal(element) {
        const modal = element.closest('.modal-dialog');
        if (modal) {
            modal.style.transform = 'scale(0.7)';
            modal.style.opacity = '0';
            setTimeout(() => {
                if (modal.parentNode) {
                    modal.parentNode.removeChild(modal);
                }
                if (this.modalContainer.children.length === 0) {
                    this.modalContainer.style.display = 'none';
                }
                
                // Mark modal as closed and process queue
                this.isModalOpen = false;
                this.processQueue();
            }, 300);
        }
    }

    processQueue() {
        if (this.modalQueue.length > 0 && !this.isModalOpen) {
            const nextModal = this.modalQueue.shift();
            setTimeout(() => {
                this.showModal(nextModal.title, nextModal.message, nextModal.type, nextModal.options);
            }, 100); // Small delay to ensure smooth transition
        }
    }

    // Convenience methods
    showSuccess(title, message, options = {}) {
        return this.showModal(title, message, 'success', options);
    }

    showError(title, message, options = {}) {
        return this.showModal(title, message, 'error', options);
    }

    showWarning(title, message, options = {}) {
        return this.showModal(title, message, 'warning', options);
    }

    showInfo(title, message, options = {}) {
        return this.showModal(title, message, 'info', options);
    }
}

// Initialize modal system
const modalSystem = new ModalSystem();

// Debug function to test modal system
window.testModal = function() {
    modalSystem.showSuccess('Test Success', 'This is a test success message');
};

// Make modalSystem globally available for debugging
window.modalSystem = modalSystem;

// Flash message to modal conversion
function convertFlashToModal() {
    const flashMessages = document.querySelectorAll('.alert, .notification-alert');
    console.log('Found flash messages:', flashMessages.length);
    console.log('Flash messages:', flashMessages);
    flashMessages.forEach(flash => {
        // Extract message text from the alert-message span
        const messageSpan = flash.querySelector('.alert-message');
        let message = '';
        if (messageSpan) {
            message = messageSpan.textContent.trim();
        } else {
            // Fallback to extracting all text nodes
            const textNodes = [];
            const walker = document.createTreeWalker(
                flash,
                NodeFilter.SHOW_TEXT,
                null,
                false
            );
            
            let node;
            while (node = walker.nextNode()) {
                textNodes.push(node.textContent.trim());
            }
            
            message = textNodes.join(' ').trim();
        }
        
        const type = flash.classList.contains('alert-danger') || flash.classList.contains('notification-danger') ? 'error' :
                    flash.classList.contains('alert-success') || flash.classList.contains('notification-success') ? 'success' :
                    flash.classList.contains('alert-warning') || flash.classList.contains('notification-warning') ? 'warning' : 'info';
        
        // Remove the flash message and any event listeners
        if (flash.parentNode) {
            flash.parentNode.removeChild(flash);
        }
        
        // Show as modal
        modalSystem.showModal(
            type === 'error' ? 'Error' : 
            type === 'success' ? 'Success' : 
            type === 'warning' ? 'Warning' : 'Information',
            message,
            type,
            { autoClose: 3000 }
        );
    });
}

// Page-specific message to modal conversion
function convertPageMessagesToModal() {
    const pageMessages = document.querySelectorAll('.alert[data-page-message="true"], .notification-alert[data-page-message="true"]');
    console.log('Found page messages:', pageMessages.length);
    pageMessages.forEach(message => {
        // Extract message text from the alert-message span
        const messageSpan = message.querySelector('.alert-message');
        let messageText = '';
        if (messageSpan) {
            messageText = messageSpan.textContent.trim();
        } else {
            // Fallback to extracting all text nodes
            const textNodes = [];
            const walker = document.createTreeWalker(
                message,
                NodeFilter.SHOW_TEXT,
                null,
                false
            );
            
            let node;
            while (node = walker.nextNode()) {
                textNodes.push(node.textContent.trim());
            }
            
            messageText = textNodes.join(' ').trim();
        }
        
        const type = message.classList.contains('alert-danger') || message.classList.contains('notification-danger') ? 'error' :
                    message.classList.contains('alert-success') || message.classList.contains('notification-success') ? 'success' :
                    message.classList.contains('alert-warning') || message.classList.contains('notification-warning') ? 'warning' : 'info';
        
        // Remove the message and any event listeners
        if (message.parentNode) {
            message.parentNode.removeChild(message);
        }
        
        // Show as modal
        modalSystem.showModal(
            type === 'error' ? 'Error' : 
            type === 'success' ? 'Success' : 
            type === 'warning' ? 'Warning' : 'Information',
            messageText,
            type,
            { autoClose: 3000 }
        );
    });
}

// Convert flash messages to modals on page load
document.addEventListener('DOMContentLoaded', function() {
    console.log('DOM Content Loaded - Modal System');
    console.log('Modal system available:', typeof modalSystem !== 'undefined');
    console.log('Modal system object:', modalSystem);
    
    // Initial conversion
    convertFlashToModal();
    convertPageMessagesToModal();
    
    // Also run after a short delay to catch any dynamically added messages
    setTimeout(() => {
        console.log('Running delayed conversion...');
        convertFlashToModal();
        convertPageMessagesToModal();
    }, 100);
    

    
    // Set up a mutation observer to catch dynamically added messages
    const observer = new MutationObserver(function(mutations) {
        mutations.forEach(function(mutation) {
            if (mutation.type === 'childList') {
                mutation.addedNodes.forEach(function(node) {
                    if (node.nodeType === 1) { // Element node
                        if (node.classList && (node.classList.contains('alert') || node.classList.contains('notification-alert'))) {
                            // Convert this specific message
                            const messageSpan = node.querySelector('.alert-message');
                            let message = '';
                            if (messageSpan) {
                                message = messageSpan.textContent.trim();
                            } else {
                                // Fallback to extracting all text nodes
                                const textNodes = [];
                                const walker = document.createTreeWalker(
                                    node,
                                    NodeFilter.SHOW_TEXT,
                                    null,
                                    false
                                );
                                
                                let textNode;
                                while (textNode = walker.nextNode()) {
                                    textNodes.push(textNode.textContent.trim());
                                }
                                
                                message = textNodes.join(' ').trim();
                            }
                            
                            const type = node.classList.contains('alert-danger') || node.classList.contains('notification-danger') ? 'error' :
                                        node.classList.contains('alert-success') || node.classList.contains('notification-success') ? 'success' :
                                        node.classList.contains('alert-warning') || node.classList.contains('notification-warning') ? 'warning' : 'info';
                            
                            // Remove the node and any event listeners
            if (node.parentNode) {
                node.parentNode.removeChild(node);
            }
                            
                            modalSystem.showModal(
                                type === 'error' ? 'Error' : 
                                type === 'success' ? 'Success' : 
                                type === 'warning' ? 'Warning' : 'Information',
                                message,
                                type,
                                { autoClose: 3000 }
                            );
                        }
                    }
                });
            }
        });
    });
    
    // Start observing
    observer.observe(document.body, {
        childList: true,
        subtree: true
    });
}); 