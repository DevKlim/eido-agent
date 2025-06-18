
document.addEventListener('DOMContentLoaded', () => {
    // Smooth scrolling for internal links
    const navLinks = document.querySelectorAll('nav a[href^="#"]');
    navLinks.forEach(link => {
        link.addEventListener('click', function (e) {
            e.preventDefault();
            const targetId = this.getAttribute('href'); // e.g., "#platform"
            const targetElement = document.querySelector(targetId); // Tries to find element with id="platform"

            if (targetElement) {
                const navbarHeight = document.querySelector('.navbar').offsetHeight;
                let elementPosition = 0;

                // For sections using .section-anchor, target the anchor for correct offset
                const sectionAnchor = targetElement.querySelector('.section-anchor') ||
                    (targetElement.classList.contains('section-anchor') ? targetElement : null) ||
                    document.querySelector(`.section-anchor[data-section-id="${targetId.substring(1)}"]`);

                if (sectionAnchor) {
                    // Get position of the section itself, not the invisible anchor if it's offset
                    const actualSection = document.getElementById(sectionAnchor.dataset.sectionId) || targetElement;
                    elementPosition = actualSection.getBoundingClientRect().top + window.pageYOffset;
                } else {
                    elementPosition = targetElement.getBoundingClientRect().top + window.pageYOffset;
                }

                const offsetPosition = elementPosition - navbarHeight;

                window.scrollTo({
                    top: offsetPosition,
                    behavior: 'smooth'
                });

                if (document.querySelector('.nav-menu.active')) {
                    document.querySelector('.nav-menu').classList.remove('active');
                    document.querySelector('.hamburger').classList.remove('active');
                }
            }
        });
    });

    const scrollElements = document.querySelectorAll('.animate-on-scroll');
    const elementInView = (el, percentageScroll = 100) => {
        const elementTop = el.getBoundingClientRect().top;
        return (
            elementTop <=
            ((window.innerHeight || document.documentElement.clientHeight) * (percentageScroll / 100))
        );
    };
    const displayScrollElement = (element) => element.classList.add('is-visible');
    const handleScrollAnimation = () => {
        scrollElements.forEach((el) => {
            const delay = parseFloat(el.style.getPropertyValue('--animation-order')) * 150;
            if (elementInView(el, 85)) {
                setTimeout(() => displayScrollElement(el), delay || 0);
            }
        });
    };
    window.addEventListener('scroll', handleScrollAnimation);
    handleScrollAnimation();

    const heroTitle = document.querySelector('.animate-hero-title');
    const heroSubtitle = document.querySelector('.animate-hero-subtitle');
    const heroButtons = document.querySelector('.hero-buttons');
    setTimeout(() => {
        if (heroTitle) { heroTitle.style.opacity = '1'; heroTitle.style.transform = 'translateY(0)'; }
        if (heroSubtitle) { heroSubtitle.style.opacity = '1'; heroSubtitle.style.transform = 'translateY(0)'; }
        if (heroButtons) { heroButtons.style.opacity = '1'; heroButtons.style.transform = 'translateY(0)'; }
    }, 100);

    const hamburger = document.querySelector('.hamburger');
    const navMenu = document.querySelector('.nav-menu');
    if (hamburger && navMenu) {
        hamburger.addEventListener('click', () => {
            hamburger.classList.toggle('active');
            navMenu.classList.toggle('active');
        });
    }

    const sectionsForNavHighlight = document.querySelectorAll('main section[id], header[id]');
    const navMenuLinks = document.querySelectorAll('.nav-menu a.nav-link');
    function navHighlighter() {
        const scrollY = window.pageYOffset;
        const navbarHeight = document.querySelector('.navbar').offsetHeight;
        let currentSectionId = "";

        sectionsForNavHighlight.forEach(current => {
            const sectionHeight = current.offsetHeight;
            const sectionTop = current.offsetTop - navbarHeight - Math.min(150, window.innerHeight * 0.25); // Adjust trigger point

            if (scrollY >= sectionTop && scrollY < sectionTop + sectionHeight) {
                currentSectionId = current.getAttribute('id');
            }
        });

        navMenuLinks.forEach(link => {
            link.classList.remove('active');
            const href = link.getAttribute('href');
            if (href && href.substring(href.indexOf('#') + 1) === currentSectionId) {
                link.classList.add('active');
            }
        });
    }
    window.addEventListener('scroll', navHighlighter);
    navHighlighter();
});