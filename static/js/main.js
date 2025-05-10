document.addEventListener('DOMContentLoaded', () => {
    // Smooth scrolling for internal links
    const navLinks = document.querySelectorAll('nav a[href^="#"]');
    navLinks.forEach(link => {
        link.addEventListener('click', function (e) {
            e.preventDefault();
            const targetId = this.getAttribute('href');
            const targetElement = document.querySelector(targetId);
            if (targetElement) {
                // Calculate position considering the fixed navbar height
                const navbarHeight = document.querySelector('.navbar').offsetHeight;
                const elementPosition = targetElement.getBoundingClientRect().top + window.pageYOffset;
                const offsetPosition = elementPosition - navbarHeight;

                window.scrollTo({
                    top: offsetPosition,
                    behavior: 'smooth'
                });
                // Close mobile menu if open
                if (document.querySelector('.nav-menu.active')) {
                    document.querySelector('.nav-menu').classList.remove('active');
                    document.querySelector('.hamburger').classList.remove('active');
                }
            }
        });
    });

    // Scroll-based animations
    const scrollElements = document.querySelectorAll('.animate-on-scroll');
    const elementInView = (el, percentageScroll = 100) => {
        const elementTop = el.getBoundingClientRect().top;
        return (
            elementTop <=
            (window.innerHeight || document.documentElement.clientHeight) * (percentageScroll / 100)
        );
    };

    const displayScrollElement = (element) => {
        element.classList.add('is-visible');
    };

    const handleScrollAnimation = () => {
        scrollElements.forEach((el) => {
            if (elementInView(el, 80)) { // Trigger when 80% of the element is visible
                displayScrollElement(el);
            }
        });
    };

    window.addEventListener('scroll', () => {
        handleScrollAnimation();
    });
    // Initial check in case elements are already in view
    handleScrollAnimation();


    // Hero section text animation (if not handled by CSS keyframes)
    // This is an example of a simple stagger animation.
    // The CSS @keyframes approach is generally preferred for simple appearances.
    // If more complex sequencing is needed, JS is better.
    const heroTitle = document.querySelector('.animate-hero-title');
    const heroSubtitle = document.querySelector('.animate-hero-subtitle');
    const heroButtons = document.querySelector('.hero-buttons');

    if (heroTitle) heroTitle.style.animationDelay = '0.2s';
    if (heroSubtitle) heroSubtitle.style.animationDelay = '0.5s';
    if (heroButtons) heroButtons.style.animationDelay = '0.8s';

    // Apply 'visible' class for CSS animations to start
    setTimeout(() => {
        if (heroTitle) heroTitle.classList.add('visible');
        if (heroSubtitle) heroSubtitle.classList.add('visible');
        if (heroButtons) heroButtons.classList.add('visible');
    }, 100); // Slight delay to ensure CSS is ready

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
    function navHighlighter() {
        let scrollY = window.pageYOffset;
        const navbarHeight = document.querySelector('.navbar').offsetHeight;

        sections.forEach(current => {
            const sectionHeight = current.offsetHeight;
            const sectionTop = current.offsetTop - navbarHeight - 50; // Adjust offset as needed
            let sectionId = current.getAttribute('id');

            let navLink = document.querySelector('.nav-menu a[href*=' + sectionId + ']');

            if (navLink) {
                if (scrollY > sectionTop && scrollY <= sectionTop + sectionHeight) {
                    document.querySelectorAll('.nav-menu a').forEach(link => link.classList.remove('active'));
                    navLink.classList.add('active');
                } else {
                    navLink.classList.remove('active');
                }
            }
        });
    }
    window.addEventListener('scroll', navHighlighter);
    // Initial call to highlight link on page load
    navHighlighter();

});