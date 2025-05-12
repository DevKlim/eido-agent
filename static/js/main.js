document.addEventListener('DOMContentLoaded', () => {
    // Smooth scrolling for internal links
    const navLinks = document.querySelectorAll('nav a[href^="#"]');
    navLinks.forEach(link => {
        link.addEventListener('click', function (e) {
            e.preventDefault();
            const targetId = this.getAttribute('href');
            const targetElement = document.querySelector(targetId);
            if (targetElement) {
                const navbarHeight = document.querySelector('.navbar').offsetHeight;
                const elementPosition = targetElement.getBoundingClientRect().top + window.pageYOffset;
                const offsetPosition = elementPosition - navbarHeight;

                window.scrollTo({
                    top: offsetPosition,
                    behavior: 'smooth'
                });

                // Close hamburger menu if open
                if (document.querySelector('.nav-menu.active')) {
                    document.querySelector('.nav-menu').classList.remove('active');
                    document.querySelector('.hamburger').classList.remove('active');
                }
            }
        });
    });

    // Scroll-based animations for elements with .animate-on-scroll
    const scrollElements = document.querySelectorAll('.animate-on-scroll');
    const elementInView = (el, percentageScroll = 100) => {
        const elementTop = el.getBoundingClientRect().top;
        return (
            elementTop <=
            ((window.innerHeight || document.documentElement.clientHeight) * (percentageScroll / 100))
        );
    };

    const displayScrollElement = (element) => {
        element.classList.add('is-visible');
    };

    const handleScrollAnimation = () => {
        scrollElements.forEach((el) => {
            // Stagger animation for feature cards if they have an animation-order style
            const delay = parseFloat(el.style.getPropertyValue('--animation-order')) * 150; // 150ms delay per order
            if (elementInView(el, 85)) { // Start animation when 85% of element is in view
                setTimeout(() => {
                    displayScrollElement(el);
                }, delay || 0);
            }
        });
    };

    window.addEventListener('scroll', handleScrollAnimation);
    handleScrollAnimation(); // Initial check on page load

    // Hero section specific text animations (triggered on load)
    const heroTitle = document.querySelector('.animate-hero-title');
    const heroSubtitle = document.querySelector('.animate-hero-subtitle');
    const heroButtons = document.querySelector('.hero-buttons');

    // Using a small timeout to ensure styles are applied before animation starts
    setTimeout(() => {
        if (heroTitle) { heroTitle.style.opacity = '1'; heroTitle.style.transform = 'translateY(0)'; }
        if (heroSubtitle) { heroSubtitle.style.opacity = '1'; heroSubtitle.style.transform = 'translateY(0)'; }
        if (heroButtons) { heroButtons.style.opacity = '1'; heroButtons.style.transform = 'translateY(0)'; }
    }, 100);


    // Hamburger menu toggle
    const hamburger = document.querySelector('.hamburger');
    const navMenu = document.querySelector('.nav-menu');

    if (hamburger && navMenu) {
        hamburger.addEventListener('click', () => {
            hamburger.classList.toggle('active');
            navMenu.classList.toggle('active');
        });
    }

    // Active Nav Link Highlighting on Scroll
    const sections = document.querySelectorAll('main section[id]');
    const navMenuLinks = document.querySelectorAll('.nav-menu a');

    function navHighlighter() {
        const scrollY = window.pageYOffset;
        const navbarHeight = document.querySelector('.navbar').offsetHeight;
        let currentSectionId = "";

        sections.forEach(current => {
            const sectionHeight = current.offsetHeight;
            // Adjust sectionTop to trigger highlight a bit earlier or later if needed
            const sectionTop = current.offsetTop - navbarHeight - Math.min(100, window.innerHeight * 0.2);

            if (scrollY >= sectionTop && scrollY < sectionTop + sectionHeight) {
                currentSectionId = current.getAttribute('id');
            }
        });

        navMenuLinks.forEach(link => {
            link.classList.remove('active');
            const href = link.getAttribute('href');
            if (href && href.includes('#') && href.substring(href.indexOf('#') + 1) === currentSectionId) {
                link.classList.add('active');
            }
        });
        // If no section is active (e.g., at the very top or bottom out of range),
        // you might want to highlight the first link or no link.
        // For simplicity, this version only highlights when a section is clearly in view.
    }

    window.addEventListener('scroll', navHighlighter);
    navHighlighter(); // Initial call
});