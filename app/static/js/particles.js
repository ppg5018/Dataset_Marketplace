// ==========================================
// MINIMAL PARTICLE SYSTEM
// Clean, subtle background effect
// ==========================================

class ParticleSystem {
    constructor() {
        this.canvas = document.getElementById('particles-canvas');
        if (!this.canvas) return;

        this.ctx = this.canvas.getContext('2d');
        this.particles = [];

        this.resize();
        this.init();
        this.animate();

        window.addEventListener('resize', () => {
            this.resize();
            this.particles = [];
            this.init();
        });
    }

    resize() {
        this.canvas.width = window.innerWidth;
        this.canvas.height = window.innerHeight;
    }

    init() {
        const particleCount = Math.min(Math.floor(window.innerWidth / 20), 50);

        for (let i = 0; i < particleCount; i++) {
            this.particles.push({
                x: Math.random() * this.canvas.width,
                y: Math.random() * this.canvas.height,
                dx: (Math.random() - 0.5) * 0.3,
                dy: (Math.random() - 0.5) * 0.3,
                size: Math.random() * 1.5 + 0.5
            });
        }
    }

    animate() {
        this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);

        this.particles.forEach(p => {
            p.x += p.dx;
            p.y += p.dy;

            if (p.x < 0 || p.x > this.canvas.width) p.dx *= -1;
            if (p.y < 0 || p.y > this.canvas.height) p.dy *= -1;

            this.ctx.globalAlpha = 0.3;
            this.ctx.fillStyle = '#20BEFF';
            this.ctx.beginPath();
            this.ctx.arc(p.x, p.y, p.size, 0, Math.PI * 2);
            this.ctx.fill();
        });

        requestAnimationFrame(() => this.animate());
    }
}

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    new ParticleSystem();

    // Theme toggle
    const savedTheme = localStorage.getItem('theme') || 'light';
    document.documentElement.setAttribute('data-theme', savedTheme);
});

// Theme toggle function
function toggleTheme() {
    const current = document.documentElement.getAttribute('data-theme') || 'light';
    const newTheme = current === 'dark' ? 'light' : 'dark';
    document.documentElement.setAttribute('data-theme', newTheme);
    localStorage.setItem('theme', newTheme);
}
