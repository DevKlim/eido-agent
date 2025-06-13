document.addEventListener('DOMContentLoaded', () => {
    const canvasContainer = document.getElementById('sentinel-node-canvas-container');
    if (!canvasContainer || typeof THREE === 'undefined') {
        console.error('3D canvas container or THREE.js not found.');
        if (canvasContainer) canvasContainer.innerHTML = "<p style='color:red; text-align:center; padding: 20px;'>Error: 3D library not loaded.</p>";
        return;
    }

    let scene, camera, renderer, clock;
    let mainDodecahedron, mainParticleSystem;

    // --- Unified Intelligence ---
    let unifiedIntelGroup, smallDodeca1, smallDodeca2, mergedObject;
    let unifiedParticles = [];
    const UNIFIED_PARTICLE_COUNT = 110;
    const MERGED_OBJECT_Y_OFFSET = -1.15;

    // --- Solutions Pipeline (Conveyor) ---
    let solutionsPipelineGroup;
    const PIPELINE_SLOT_COUNT = 4;
    const PIPELINE_SLOT_SPACING = 2.6;
    const PIPELINE_OBJECT_Y = 0;
    const PIPELINE_OBJECT_Y_SPAWN_OFFSET = 1.6;
    const PIPELINE_MOVE_SPEED = 1.8;
    let pipelineSlots = [];
    let nextSpawnTime = 0;

    let sections = [];
    let currentSectionId = 'hero';
    const targetNodeY = { value: 0 };
    let lastScrollY = window.scrollY;

    const ACTIVE_OPACITY = 0.97; // Slightly higher active opacity

    const activeNodeProps = {
        scale: 1.0, emissiveIntensity: 0.5,
        color: new THREE.Color(0xffffff),
        emissive: new THREE.Color(0x00B8D9),
        rotationSpeedX: 0.001, rotationSpeedY: 0.001,
        currentOpacity: 0.0, targetOpacity: 0.0
    };

    // Modern Gradient Palettes: [Primary/Top, Secondary/Bottom]
    const sectionAnimationConfigs = {
        'hero': {
            name: 'hero', vertexGradientColors: ['#00E5FF', '#0069C0'], // Vibrant Cyan to Deep Blue
            targetEmissiveHex: 0x60EFFF, targetEmissiveIntensity: 0.7,
            targetScale: 1.65, targetOpacity: ACTIVE_OPACITY, wireframe: false,
            targetRotationSpeed: { x: 0.0006, y: 0.0011 }, animationType: 'pulseAndGlow'
        },
        'platform': {
            name: 'platform', animationType: 'unifiedIntelligence', targetOpacity: 0.0,
            vertexGradientColors: ['#00B0FF', '#004C8C'], targetEmissiveHex: 0x00A0C0, targetEmissiveIntensity: 0.5,
            targetScale: 1.3, targetRotationSpeed: { x: 0.001, y: 0.0015 }
        },
        'solutions': {
            name: 'solutions', animationType: 'solutionsPipeline', targetOpacity: 0.0,
            vertexGradientColors: ['#40C4FF', '#005CB2'], targetEmissiveHex: 0x29B6F6, targetEmissiveIntensity: 0.6,
            targetScale: 1.4, targetRotationSpeed: { x: 0.0008, y: 0.0013 }
        },
        'collaboration': {
            name: 'collaboration', vertexGradientColors: ['#80D8FF', '#2675B2'], // Lighter Blue
            targetEmissiveHex: 0xA0E0FF, targetEmissiveIntensity: 0.55,
            targetScale: 1.25, targetOpacity: ACTIVE_OPACITY, wireframe: true,
            targetRotationSpeed: { x: 0.0011, y: 0.0008 }, animationType: 'breathingWireframePulse'
        },
        'capabilities': {
            name: 'capabilities', vertexGradientColors: ['#00CFD1', '#005F6B'], // Teal/Cyan
            targetEmissiveHex: 0x50DFE1, targetEmissiveIntensity: 0.65,
            targetScale: 1.35, targetOpacity: ACTIVE_OPACITY, wireframe: false,
            targetRotationSpeed: { x: 0.0009, y: 0.001 }, animationType: 'dynamicProcessing'
        }
    };
    let currentTargetConfig = sectionAnimationConfigs.hero;

    function applyVertexGradient(geometry, color1, color2, axis = 'y') { /* ... (same) ... */
        geometry.computeBoundingBox();
        const bbox = geometry.boundingBox;
        const colors = [];
        const colorObj1 = new THREE.Color(color1);
        const colorObj2 = new THREE.Color(color2);

        const positionAttribute = geometry.attributes.position;
        for (let i = 0; i < positionAttribute.count; i++) {
            const vertex = new THREE.Vector3().fromBufferAttribute(positionAttribute, i);
            let alpha;
            if (axis === 'y') {
                alpha = (vertex.y - bbox.min.y) / (bbox.max.y - bbox.min.y);
            } else if (axis === 'x') {
                alpha = (vertex.x - bbox.min.x) / (bbox.max.x - bbox.min.x);
            } else { // axis === 'z'
                alpha = (vertex.z - bbox.min.z) / (bbox.max.z - bbox.min.z);
            }
            alpha = Math.max(0, Math.min(1, alpha));
            const interpolatedColor = colorObj1.clone().lerp(colorObj2, alpha);
            colors.push(interpolatedColor.r, interpolatedColor.g, interpolatedColor.b);
        }
        geometry.setAttribute('color', new THREE.Float32BufferAttribute(colors, 3));
    }


    function init() { /* ... (Lighting slightly adjusted for sleeker feel) ... */
        scene = new THREE.Scene();
        clock = new THREE.Clock();
        camera = new THREE.PerspectiveCamera(50, canvasContainer.clientWidth / canvasContainer.clientHeight, 0.1, 100);
        camera.position.set(0, 0.1, 8.2); // Slight elevation, further back
        camera.lookAt(0, 0, 0);

        renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
        renderer.setSize(canvasContainer.clientWidth, canvasContainer.clientHeight);
        renderer.setPixelRatio(window.devicePixelRatio);
        renderer.outputEncoding = THREE.sRGBEncoding;
        canvasContainer.appendChild(renderer.domElement);

        scene.add(new THREE.AmbientLight(0xE0F2FE, 0.5)); // Lighter ambient
        const hemisphereLight = new THREE.HemisphereLight(0xC0E0FF, 0x405060, 1.1);
        scene.add(hemisphereLight);
        const directionalLight1 = new THREE.DirectionalLight(0xffffff, 1.0);
        directionalLight1.position.set(4, 6, 5);
        scene.add(directionalLight1);
        const directionalLight2 = new THREE.DirectionalLight(0xB0D0FF, 0.7);
        directionalLight2.position.set(-3, -1, 2);
        scene.add(directionalLight2);

        const dodecaGeometry = new THREE.DodecahedronGeometry(1.25, 0); // Slightly larger main
        applyVertexGradient(dodecaGeometry, currentTargetConfig.vertexGradientColors[0], currentTargetConfig.vertexGradientColors[1]);

        const dodecaMaterial = new THREE.MeshStandardMaterial({
            vertexColors: true,
            emissive: activeNodeProps.emissive,
            emissiveIntensity: activeNodeProps.emissiveIntensity,
            metalness: 0.1, roughness: 0.75, // More matte, less metallic for gradients
            transparent: true, opacity: activeNodeProps.currentOpacity,
            wireframe: currentTargetConfig.wireframe,
            polygonOffset: true, polygonOffsetFactor: 1, polygonOffsetUnits: 1 // Helps with wireframe rendering over solid
        });
        mainDodecahedron = new THREE.Mesh(dodecaGeometry, dodecaMaterial);
        scene.add(mainDodecahedron);
        mainParticleSystem = createGenericParticleSystem(0xA0DFFF, 280, 1.0, 0.038); // Slightly adjusted
        mainDodecahedron.add(mainParticleSystem);

        initUnifiedIntelligenceGroup();
        initSolutionsPipelineGroup();

        collectSectionsData();
        activeNodeProps.targetOpacity = sectionAnimationConfigs[currentSectionId]?.targetOpacity === 0 ? 0 : ACTIVE_OPACITY;
        activeNodeProps.currentOpacity = activeNodeProps.targetOpacity;
        mainDodecahedron.material.opacity = activeNodeProps.currentOpacity;
        applyTargetConfigInstantly(currentTargetConfig);

        updateNodeVerticalPosition();
        handleSectionTransition(null, currentSectionId);

        window.addEventListener('resize', onWindowResize, false);
        window.addEventListener('scroll', onWindowScroll, { passive: true });
        animate();
    }

    function createGenericParticleSystem(color, count, baseRadius, particleSize) { /* ... (same) ... */
        const positions = [];
        for (let i = 0; i < count; i++) {
            const u = Math.random(); const v = Math.random();
            const theta = 2 * Math.PI * u; const phi = Math.acos(2 * v - 1);
            const r = Math.cbrt(Math.random()) * baseRadius;
            positions.push(r * Math.sin(phi) * Math.cos(theta), r * Math.sin(phi) * Math.sin(theta), r * Math.cos(phi));
        }
        const pGeometry = new THREE.BufferGeometry();
        pGeometry.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
        const pMaterial = new THREE.PointsMaterial({
            size: particleSize, color: color, transparent: true, opacity: 0.75,
            blending: THREE.AdditiveBlending, sizeAttenuation: true, depthWrite: false
        });
        const system = new THREE.Points(pGeometry, pMaterial);
        system.visible = false;
        return system;
    }

    function initUnifiedIntelligenceGroup() {
        unifiedIntelGroup = new THREE.Group();
        unifiedIntelGroup.userData.targetOpacity = ACTIVE_OPACITY;
        unifiedIntelGroup.userData.currentOpacity = 0;

        const smallGeom = new THREE.DodecahedronGeometry(0.8, 0); // Consistent size
        const smallGrad1 = ['#00E0FF', '#0070A0']; // Cyan theme
        const smallGrad2 = ['#40C4FF', '#005CB2']; // Light Blue theme
        applyVertexGradient(smallGeom, smallGrad1[0], smallGrad1[1]);

        const matProps = { vertexColors: true, metalness: 0.15, roughness: 0.7, emissiveIntensity: 0.45 };
        smallDodeca1 = new THREE.Mesh(smallGeom.clone(), new THREE.MeshStandardMaterial({ emissive: 0x30D0EF, ...matProps }));
        applyVertexGradient(smallDodeca1.geometry, smallGrad1[0], smallGrad1[1]); // Apply gradient to cloned geo

        smallDodeca2 = new THREE.Mesh(smallGeom.clone(), new THREE.MeshStandardMaterial({ emissive: 0x50B0DF, ...matProps }));
        applyVertexGradient(smallDodeca2.geometry, smallGrad2[0], smallGrad2[1]);

        smallDodeca1.userData.baseEmissiveIntensity = matProps.emissiveIntensity;
        smallDodeca2.userData.baseEmissiveIntensity = matProps.emissiveIntensity;
        [smallDodeca1, smallDodeca2].forEach(d => { d.material.transparent = true; d.material.opacity = 0; });

        const mergedGeom = new THREE.SphereGeometry(1.2, 32, 24);
        applyVertexGradient(mergedGeom, '#A0F0FF', '#3080A0', 'y'); // Lighter gradient for feeder
        mergedObject = new THREE.Mesh(mergedGeom, new THREE.MeshStandardMaterial({
            vertexColors: true,
            emissive: 0x70C0D0, emissiveIntensity: 0.6, metalness: 0.05, roughness: 0.8,
            transparent: true, opacity: 0
        }));
        mergedObject.position.y = MERGED_OBJECT_Y_OFFSET;
        mergedObject.userData.baseEmissiveIntensity = 0.6;

        unifiedIntelGroup.add(smallDodeca1, smallDodeca2, mergedObject);

        const particleMaterial = new THREE.SpriteMaterial({ /* ... (same) ... */
            map: new THREE.TextureLoader().load('/static/images/particle_glow.png'),
            color: 0xE0FFFF, blending: THREE.AdditiveBlending, transparent: true, opacity: 0.95, depthWrite: false,
            sizeAttenuation: true
        });

        for (let i = 0; i < UNIFIED_PARTICLE_COUNT * 2; i++) { /* ... (particle scale slightly adjusted) ... */
            const particle = new THREE.Sprite(particleMaterial.clone());
            const scale = Math.random() * 0.11 + 0.055;
            particle.scale.set(scale, scale, scale);
            particle.material.opacity = 0;
            particle.userData = {
                source: mergedObject,
                target: (i < UNIFIED_PARTICLE_COUNT) ? smallDodeca1 : smallDodeca2,
                velocity: new THREE.Vector3(),
                life: 0,
                maxLife: Math.random() * 1.7 + 0.9,
                isActive: false,
                delay: Math.random() * 0.7
            };
            resetUnifiedParticle(particle, true);
            unifiedParticles.push(particle);
            unifiedIntelGroup.add(particle);
        }

        unifiedIntelGroup.visible = false;
        scene.add(unifiedIntelGroup);
    }

    function resetUnifiedParticle(particle, isInitial = false) { /* ... (same) ... */
        const sourcePos = particle.userData.source.position;
        const emitterRadius = particle.userData.source.geometry.parameters.radius * 0.9;

        const phi = Math.acos(-1 + (2 * Math.random()));
        const theta = Math.sqrt(UNIFIED_PARTICLE_COUNT * 2 * Math.PI) * phi * Math.random();
        particle.position.set(
            sourcePos.x + emitterRadius * Math.sin(phi) * Math.cos(theta),
            sourcePos.y + emitterRadius * Math.sin(phi) * Math.sin(theta),
            sourcePos.z + emitterRadius * Math.cos(phi)
        );

        particle.userData.life = isInitial ? -particle.userData.delay : 0;
        particle.userData.isActive = false;
        particle.visible = false;
    }

    function initSolutionsPipelineGroup() { /* ... (same) ... */
        solutionsPipelineGroup = new THREE.Group();
        solutionsPipelineGroup.userData.targetOpacity = ACTIVE_OPACITY;
        solutionsPipelineGroup.userData.currentOpacity = 0;
        solutionsPipelineGroup.visible = false;

        for (let i = 0; i < PIPELINE_SLOT_COUNT; i++) {
            pipelineSlots.push(null);
        }
        scene.add(solutionsPipelineGroup);
    }

    function spawnNewPipelineObject(slotIndexToFill) { /* ... (sleeker gradient choices) ... */
        if (pipelineSlots[slotIndexToFill]) {
            solutionsPipelineGroup.remove(pipelineSlots[slotIndexToFill]);
            pipelineSlots[slotIndexToFill].geometry.dispose();
            pipelineSlots[slotIndexToFill].material.dispose();
        }

        const geometries = [ // Modernized gradients
            { type: THREE.BoxGeometry, args: [1.0, 1.0, 1.0], scale: 1, gradCols: ['#4FC3F7', '#0277BD'] }, // Light Blue to Dark Blue
            { type: THREE.SphereGeometry, args: [0.75, 24, 12], scale: 1, gradCols: ['#26C6DA', '#006064'] }, // Cyan to Dark Teal
            { type: THREE.IcosahedronGeometry, args: [0.85, 0], scale: 1, gradCols: ['#81D4FA', '#01579B'] }, // Sky Blue to Navy
            { type: THREE.DodecahedronGeometry, args: [0.9, 0], scale: 1, gradCols: ['#4DD0E1', '#004D40'] }, // Teal to Dark Green/Teal
            { type: THREE.TorusKnotGeometry, args: [0.65, 0.18, 60, 10], scale: 1.1, gradCols: ['#B2EBF2', '#00ACC1'] } // Light Cyan to Medium Cyan
        ];
        const choice = geometries[Math.floor(Math.random() * geometries.length)];
        const geom = new choice.type(...choice.args);
        applyVertexGradient(geom, choice.gradCols[0], choice.gradCols[1]);

        const mesh = new THREE.Mesh(
            geom,
            new THREE.MeshStandardMaterial({
                vertexColors: true,
                emissive: new THREE.Color(choice.gradCols[1]).lerp(new THREE.Color(0xffffff), 0.1), // Emissive based on darker gradient color
                emissiveIntensity: 0.15,
                metalness: 0.05, roughness: 0.8, transparent: true, opacity: 0,
            })
        );

        mesh.userData.baseScale = choice.scale;
        mesh.scale.setScalar(choice.scale * 0.01);
        mesh.position.set(
            (PIPELINE_SLOT_COUNT / 2) * PIPELINE_SLOT_SPACING + PIPELINE_SLOT_SPACING * 1.5, // Spawn further right
            PIPELINE_OBJECT_Y + PIPELINE_OBJECT_Y_SPAWN_OFFSET,
            (Math.random() - 0.5) * 0.15 // Very slight Z spread
        );
        mesh.rotation.set(Math.random() * 0.5 - 0.25, Math.random() * Math.PI, Math.random() * 0.5 - 0.25); // More controlled random rotation
        mesh.userData.age = 0;
        mesh.userData.targetSlotX = ((PIPELINE_SLOT_COUNT - 1) / 2 - slotIndexToFill) * -PIPELINE_SLOT_SPACING;

        solutionsPipelineGroup.add(mesh);
        pipelineSlots[slotIndexToFill] = mesh;
    }

    function applyTargetConfigInstantly(config) { /* ... (Vertex gradient update moved to updateNodeVerticalPosition) ... */
        if (!mainDodecahedron || !config) return;
        activeNodeProps.emissive.set(config.targetEmissiveHex);
        activeNodeProps.emissiveIntensity = config.targetEmissiveIntensity;
        activeNodeProps.targetOpacity = config.targetOpacity;
        activeNodeProps.scale = config.targetScale;
        activeNodeProps.rotationSpeedX = config.targetRotationSpeed.x;
        activeNodeProps.rotationSpeedY = config.targetRotationSpeed.y;

        const mat = mainDodecahedron.material;
        mat.emissive.set(activeNodeProps.emissive);
        mat.emissiveIntensity = activeNodeProps.emissiveIntensity;
        mat.wireframe = config.wireframe;
        mainDodecahedron.scale.set(activeNodeProps.scale, activeNodeProps.scale, activeNodeProps.scale);
    }

    function collectSectionsData() { /* ... (same) ... */
        sections = [];
        const sectionElements = document.querySelectorAll('main section[id], header[id]');
        sectionElements.forEach(el => {
            sections.push({ id: el.id, el: el, top: el.offsetTop, height: el.offsetHeight });
        });
        sections.sort((a, b) => a.top - b.top);
    }

    function updateNodeVerticalPosition() { /* ... (same, includes main dodeca gradient update) ... */
        let newSectionId = currentSectionId;
        const viewportCenterY = window.scrollY + window.innerHeight / 2;
        const navBarHeight = document.querySelector('.navbar')?.offsetHeight || 65;
        let bestMatch = { id: null, sectionCenterDelta: Infinity };

        for (const section of sections) {
            const sectionTopAbsolute = section.top;
            const sectionBottomAbsolute = section.top + section.height;
            const sectionCenterAbsolute = sectionTopAbsolute + section.height / 2;
            const deltaFromViewportCenter = Math.abs(sectionCenterAbsolute - viewportCenterY);

            const sectionTopOnScreen = sectionTopAbsolute - window.scrollY;
            const sectionBottomOnScreen = sectionBottomAbsolute - window.scrollY;
            const visibleTop = Math.max(sectionTopOnScreen, navBarHeight);
            const visibleBottom = Math.min(sectionBottomOnScreen, window.innerHeight);
            const visibleHeight = Math.max(0, visibleBottom - visibleTop);

            if (visibleHeight > window.innerHeight * 0.1 && deltaFromViewportCenter < bestMatch.sectionCenterDelta) {
                bestMatch.sectionCenterDelta = deltaFromViewportCenter;
                bestMatch.id = section.id;
            }
        }

        if (bestMatch.id) newSectionId = bestMatch.id;
        else if (sections.length > 0 && window.scrollY < sections[0].top) newSectionId = sections[0].id;
        else if (sections.length > 0 && (window.scrollY + window.innerHeight) > (sections[sections.length - 1].top + sections[sections.length - 1].height)) newSectionId = sections[sections.length - 1].id;

        if (currentSectionId !== newSectionId && newSectionId !== null) {
            handleSectionTransition(currentSectionId, newSectionId);
            currentSectionId = newSectionId;
            currentTargetConfig = sectionAnimationConfigs[currentSectionId] || sectionAnimationConfigs.platform;
            if (mainDodecahedron.visible && currentTargetConfig.vertexGradientColors) { // Update main dodeca gradient if it's now visible
                applyVertexGradient(mainDodecahedron.geometry, currentTargetConfig.vertexGradientColors[0], currentTargetConfig.vertexGradientColors[1]);
                mainDodecahedron.geometry.attributes.color.needsUpdate = true;
            }
        }
        targetNodeY.value = 0;
        if (currentSectionId === 'hero' && window.scrollY < window.innerHeight * 0.2) targetNodeY.value = 0.25; // Hero slightly higher
    }

    function handleSectionTransition(oldSectionId, newSectionId) { /* ... (robust unloading for pipeline) ... */
        const oldConfig = sectionAnimationConfigs[oldSectionId];
        const newConfig = sectionAnimationConfigs[newSectionId];

        if (oldConfig) {
            if (oldConfig.animationType === 'unifiedIntelligence') {
                unifiedIntelGroup.userData.targetOpacity = 0;
                // Particles and objects will fade with group and reset when group becomes active again
            } else if (oldConfig.animationType === 'solutionsPipeline') {
                solutionsPipelineGroup.userData.targetOpacity = 0;
                // Objects will fade with group; actual removal happens if group becomes invisible for long or targetOpacity remains 0
                // Or more proactively:
                pipelineSlots.forEach((mesh, index) => { // Start fade out for all current pipeline items
                    if (mesh) mesh.userData.isFadingOut = true;
                });

            } else {
                activeNodeProps.targetOpacity = 0;
            }
        }

        if (newConfig) {
            if (newConfig.animationType === 'unifiedIntelligence') {
                unifiedIntelGroup.visible = true;
                unifiedIntelGroup.userData.targetOpacity = ACTIVE_OPACITY;
                // Reset states for fresh animation
                smallDodeca1.position.set(0, 0, 0); smallDodeca1.scale.set(0.01, 0.01, 0.01); smallDodeca1.material.opacity = 0;
                smallDodeca2.position.set(0, 0, 0); smallDodeca2.scale.set(0.01, 0.01, 0.01); smallDodeca2.material.opacity = 0;
                mergedObject.position.y = MERGED_OBJECT_Y_OFFSET;
                mergedObject.scale.set(0.01, 0.01, 0.01); mergedObject.material.opacity = 0;
                unifiedParticles.forEach(p => resetUnifiedParticle(p, true));
            } else if (newConfig.animationType === 'solutionsPipeline') {
                solutionsPipelineGroup.visible = true;
                solutionsPipelineGroup.userData.targetOpacity = ACTIVE_OPACITY;
                nextSpawnTime = clock.getElapsedTime() + 0.1; // Start spawning shortly after transition
                // Ensure slots are clean before starting new pipeline
                pipelineSlots.forEach((mesh, index) => {
                    if (mesh) {
                        solutionsPipelineGroup.remove(mesh);
                        mesh.geometry.dispose(); mesh.material.dispose();
                        pipelineSlots[index] = null;
                    }
                });
            } else {
                mainDodecahedron.visible = true;
                activeNodeProps.targetOpacity = newConfig.targetOpacity;
                mainDodecahedron.material.wireframe = newConfig.wireframe || false;
                if (mainDodecahedron.material.wireframe) mainDodecahedron.material.needsUpdate = true;
            }
            if (mainParticleSystem) mainParticleSystem.visible = mainDodecahedron.visible && (newConfig.animationType === 'internalStreams' || newConfig.animationType === 'dynamicProcessing');
        }
    }

    function onWindowScroll() { /* ... (same) ... */
        if (Math.abs(window.scrollY - lastScrollY) > 0.5) { // More sensitive scroll
            updateNodeVerticalPosition(); lastScrollY = window.scrollY;
        }
    }
    function onWindowResize() { /* ... (same) ... */
        if (!renderer || !camera || !canvasContainer) return;
        const newWidth = canvasContainer.clientWidth; const newHeight = canvasContainer.clientHeight;
        if (newWidth > 0 && newHeight > 0) {
            camera.aspect = newWidth / newHeight; camera.updateProjectionMatrix();
            renderer.setSize(newWidth, newHeight);
        }
        collectSectionsData(); updateNodeVerticalPosition();
    }
    function lerp(start, end, t) { return start * (1 - t) + end * t; }

    function animateUnifiedIntelligence(time, deltaTime) { /* ... (ensure opacity respects group opacity) ... */
        if (!unifiedIntelGroup.visible && unifiedIntelGroup.userData.currentOpacity < 0.005) return;
        unifiedIntelGroup.userData.currentOpacity = lerp(unifiedIntelGroup.userData.currentOpacity, unifiedIntelGroup.userData.targetOpacity, deltaTime * 3.0);
        const groupOpacity = unifiedIntelGroup.userData.currentOpacity;

        const spread = 1.35; const verticalOffset = 0.65; const animSpeed = 2.2;

        [smallDodeca1, smallDodeca2].forEach(obj => {
            obj.material.opacity = groupOpacity; // Link child opacity to group's current opacity
            obj.scale.lerp(new THREE.Vector3(1, 1, 1).multiplyScalar(groupOpacity > 0.05 ? 1 : 0.01), deltaTime * animSpeed);
        });
        mergedObject.material.opacity = groupOpacity;
        mergedObject.scale.lerp(new THREE.Vector3(1, 1, 1).multiplyScalar(groupOpacity > 0.05 ? 1 : 0.01), deltaTime * animSpeed * 0.8);


        smallDodeca1.position.lerp(new THREE.Vector3(-spread, verticalOffset, 0), deltaTime * animSpeed);
        smallDodeca2.position.lerp(new THREE.Vector3(spread, verticalOffset, 0), deltaTime * animSpeed);

        smallDodeca1.rotation.y += 0.0075; smallDodeca1.rotation.x += 0.0028;
        smallDodeca2.rotation.y -= 0.0075; smallDodeca2.rotation.x -= 0.0028;
        mergedObject.rotation.y += 0.003;

        let absorbedThisFrame = 0;
        unifiedParticles.forEach(particle => {
            particle.material.opacity = groupOpacity * 0.9;
            particle.userData.life += deltaTime;
            if (particle.userData.life < 0) return;

            if (!particle.userData.isActive) particle.userData.isActive = true;
            if (!particle.visible && groupOpacity > 0.1) particle.visible = true;
            else if (particle.visible && groupOpacity < 0.05) particle.visible = false; // Hide if group almost invisible
            if (!particle.visible) return;

            const targetWorldPos = new THREE.Vector3(); // Target is smallDodeca 1 or 2
            particle.userData.target.getWorldPosition(targetWorldPos);

            const direction = targetWorldPos.clone().sub(particle.position).normalize();
            const moveSpeed = 3.2 + Math.random() * 0.7;
            particle.position.add(direction.multiplyScalar(moveSpeed * deltaTime));

            if (particle.position.distanceTo(targetWorldPos) < 0.35 || particle.userData.life > particle.userData.maxLife) {
                resetUnifiedParticle(particle);
                absorbedThisFrame++;
                const targetMat = particle.userData.target.material;
                targetMat.emissiveIntensity = particle.userData.target.userData.baseEmissiveIntensity + 0.7; // Brighter pulse
            }
        });

        [smallDodeca1, smallDodeca2].forEach(d => {
            d.material.emissiveIntensity = lerp(d.material.emissiveIntensity, d.userData.baseEmissiveIntensity, deltaTime * 7); // Faster decay
        });

        if (groupOpacity < 0.005 && unifiedIntelGroup.userData.targetOpacity < 0.005) {
            unifiedIntelGroup.visible = false;
        }
    }

    function animateSolutionsPipeline(time, deltaTime) { /* ... (opacity tied to group, more robust exit fade) ... */
        if (!solutionsPipelineGroup.visible && solutionsPipelineGroup.userData.currentOpacity < 0.005) return;

        solutionsPipelineGroup.userData.currentOpacity = lerp(solutionsPipelineGroup.userData.currentOpacity, solutionsPipelineGroup.userData.targetOpacity, deltaTime * 3.0);
        const groupOpacity = solutionsPipelineGroup.userData.currentOpacity;

        if (groupOpacity > 0.005 && !solutionsPipelineGroup.visible) solutionsPipelineGroup.visible = true;

        if (time >= nextSpawnTime && groupOpacity > 0.7) {
            let emptySlotIndex = -1;
            for (let i = PIPELINE_SLOT_COUNT - 1; i >= 0; i--) {
                if (!pipelineSlots[i]) {
                    emptySlotIndex = i;
                    break;
                }
            }
            if (emptySlotIndex !== -1) {
                spawnNewPipelineObject(emptySlotIndex);
                nextSpawnTime = time + (PIPELINE_SLOT_SPACING / PIPELINE_MOVE_SPEED) * (0.75 + Math.random() * 0.35); // Slightly more varied interval
            }
        }

        for (let i = 0; i < PIPELINE_SLOT_COUNT; i++) {
            const mesh = pipelineSlots[i];
            if (mesh) {
                mesh.userData.age += deltaTime;
                mesh.position.x -= PIPELINE_MOVE_SPEED * deltaTime;

                const landDuration = 1.8;
                let yPos = PIPELINE_OBJECT_Y;
                let objectOpacityTarget = groupOpacity * 0.97; // Base on group, then modify
                let scaleScalar = mesh.userData.baseScale;

                if (mesh.userData.age < landDuration) {
                    const landProgress = Math.min(1.0, mesh.userData.age / landDuration);
                    const easeOutQuad = t => t * (2 - t);
                    const easeInOutSine = t => -(Math.cos(Math.PI * t) - 1) / 2;

                    yPos = lerp(PIPELINE_OBJECT_Y + PIPELINE_OBJECT_Y_SPAWN_OFFSET, PIPELINE_OBJECT_Y, easeOutQuad(landProgress));
                    objectOpacityTarget *= easeInOutSine(landProgress); // Fade in with group opacity constraint
                    scaleScalar = lerp(0.01 * mesh.userData.baseScale, mesh.userData.baseScale, easeInOutSine(landProgress));
                }

                const visualEndX = - (PIPELINE_SLOT_COUNT / 2 + 0.5) * PIPELINE_SLOT_SPACING;
                const fadeOutStartPos = visualEndX + PIPELINE_SLOT_SPACING * 1.5; // Start fading earlier
                if (mesh.position.x < fadeOutStartPos) {
                    const fadeProgress = Math.max(0, Math.min(1, (mesh.position.x - visualEndX) / (fadeOutStartPos - visualEndX)));
                    objectOpacityTarget *= fadeProgress;
                }
                // If the main group is fading out, accelerate object fade
                if (mesh.userData.isFadingOut) {
                    objectOpacityTarget = lerp(mesh.material.opacity, 0, deltaTime * 5); // Fast fade if group is exiting
                }


                mesh.position.y = yPos;
                mesh.material.opacity = objectOpacityTarget; // Apply calculated opacity
                mesh.scale.setScalar(scaleScalar);

                mesh.rotation.x += deltaTime * 0.28 * (i % 2 === 0 ? 0.9 : -1.15); // Slower, smoother rotation
                mesh.rotation.y += deltaTime * 0.32 * (i % 2 === 0 ? 1.1 : -0.95);

                if (mesh.position.x < - (PIPELINE_SLOT_COUNT / 2 + 2.5) * PIPELINE_SLOT_SPACING) {
                    solutionsPipelineGroup.remove(mesh);
                    mesh.geometry.dispose(); mesh.material.dispose();
                    pipelineSlots[i] = null;
                }
            }
        }
        if (groupOpacity < 0.005 && solutionsPipelineGroup.userData.targetOpacity < 0.005) {
            solutionsPipelineGroup.visible = false;
            // Final cleanup when group becomes invisible after fading out
            pipelineSlots.forEach((mesh, index) => {
                if (mesh) {
                    solutionsPipelineGroup.remove(mesh);
                    mesh.geometry.dispose(); mesh.material.dispose();
                    pipelineSlots[index] = null;
                }
            });
        }
    }

    function handleMainDodecaAnimation(config, time, deltaTime) { /* ... (opacity of particles tied to main dodeca opacity) ... */
        if (!mainDodecahedron.visible || !config || activeNodeProps.currentOpacity < 0.005) {
            if (mainParticleSystem) mainParticleSystem.visible = false;
            return;
        }

        mainDodecahedron.rotation.x += activeNodeProps.rotationSpeedX * deltaTime * 50; // Slightly slower base rotation
        mainDodecahedron.rotation.y += activeNodeProps.rotationSpeedY * deltaTime * 50;

        const showMainParticles = config.animationType === 'internalStreams' || config.animationType === 'dynamicProcessing';
        if (mainParticleSystem) {
            mainParticleSystem.visible = showMainParticles && (activeNodeProps.currentOpacity > 0.1);
            if (mainParticleSystem.material) mainParticleSystem.material.opacity = activeNodeProps.currentOpacity * 0.75; // Link particle opacity
        }

        switch (config.animationType) {
            case 'pulseAndGlow':
                const pulseFactor = (Math.sin(time * 1.9) + 1) / 2;
                activeNodeProps.emissiveIntensity = lerp(activeNodeProps.emissiveIntensity, config.targetEmissiveIntensity * (0.45 + pulseFactor * 0.85), 0.13);
                break;
            case 'breathingWireframePulse':
                const breathFactor = (Math.sin(time * 1.7) + 1) / 2;
                activeNodeProps.scale = lerp(activeNodeProps.scale, config.targetScale * (0.93 + breathFactor * 0.14), 0.09);
                activeNodeProps.emissiveIntensity = lerp(activeNodeProps.emissiveIntensity, config.targetEmissiveIntensity * (0.25 + breathFactor * 1.5), 0.1);
                if (mainDodecahedron.material.wireframe) mainDodecahedron.material.wireframeLinewidth = lerp(mainDodecahedron.material.wireframeLinewidth || 1, 0.4 + breathFactor * 2.2, 0.11);
                break;
            case 'dynamicProcessing':
                if (mainParticleSystem && mainParticleSystem.visible) {
                    mainParticleSystem.rotation.x += 0.0033; mainParticleSystem.rotation.z -= 0.0022;
                    const positions = mainParticleSystem.geometry.attributes.position.array;
                    const scaleFactor = mainDodecahedron.scale.x * 0.93;
                    for (let i = 0; i < positions.length; i += 3) {
                        positions[i] += (Math.random() - 0.5) * 0.028; positions[i + 1] += (Math.random() - 0.5) * 0.028; positions[i + 2] += (Math.random() - 0.5) * 0.028;
                        const distSq = positions[i] * positions[i] + positions[i + 1] * positions[i + 1] + positions[i + 2] * positions[i + 2];
                        if (distSq > scaleFactor * scaleFactor) {
                            const L = Math.sqrt(distSq);
                            positions[i] = (positions[i] / L) * scaleFactor * (0.88 + Math.random() * 0.08);
                            positions[i + 1] = (positions[i + 1] / L) * scaleFactor * (0.88 + Math.random() * 0.08);
                            positions[i + 2] = (positions[i + 2] / L) * scaleFactor * (0.88 + Math.random() * 0.08);
                        }
                    }
                    mainParticleSystem.geometry.attributes.position.needsUpdate = true;
                }
                break;
            case 'internalStreams':
                if (mainParticleSystem && mainParticleSystem.visible) {
                    mainParticleSystem.rotation.y += 0.004;
                }
                break;
            default:
                if (config.targetEmissiveIntensity) {
                    activeNodeProps.emissiveIntensity = lerp(activeNodeProps.emissiveIntensity, config.targetEmissiveIntensity * (0.8 + Math.sin(time * 1.9) * 0.2), 0.11);
                }
                break;
        }
    }

    function animate() { /* ... (Main animate loop now primarily handles opacity lerping for groups/main dodeca, and dispatches to specific anim handlers) ... */
        requestAnimationFrame(animate);
        if (!currentTargetConfig) return;

        const deltaTime = Math.min(clock.getDelta(), 0.1);
        const time = clock.getElapsedTime();
        const lerpFactor = Math.min(deltaTime * 5.5, 1); // Global lerp for properties

        // --- Handle Main Dodecahedron Visibility & Properties ---
        activeNodeProps.currentOpacity = lerp(activeNodeProps.currentOpacity, activeNodeProps.targetOpacity, deltaTime * 4.0); // Control main dodeca opacity
        mainDodecahedron.material.opacity = activeNodeProps.currentOpacity;

        if (mainParticleSystem && mainParticleSystem.material) { // Link main particle system opacity
            mainParticleSystem.material.opacity = activeNodeProps.currentOpacity > 0 ? activeNodeProps.currentOpacity * 0.75 : 0;
        }


        if (activeNodeProps.currentOpacity > 0.001) {
            if (!mainDodecahedron.visible) mainDodecahedron.visible = true;

            if (currentTargetConfig.targetOpacity > 0) { // Main dodeca is the primary focus
                activeNodeProps.emissive.lerp(new THREE.Color(currentTargetConfig.targetEmissiveHex), lerpFactor);

                if (currentTargetConfig.animationType !== 'pulseAndGlow' &&
                    currentTargetConfig.animationType !== 'breathingWireframePulse' &&
                    currentTargetConfig.animationType !== 'subtleShimmer') {
                    activeNodeProps.emissiveIntensity = lerp(activeNodeProps.emissiveIntensity, currentTargetConfig.targetEmissiveIntensity, lerpFactor);
                }
                if (currentTargetConfig.animationType !== 'breathingWireframePulse') {
                    activeNodeProps.scale = lerp(activeNodeProps.scale, currentTargetConfig.targetScale, lerpFactor);
                }
                activeNodeProps.rotationSpeedX = lerp(activeNodeProps.rotationSpeedX, currentTargetConfig.targetRotationSpeed.x, lerpFactor);
                activeNodeProps.rotationSpeedY = lerp(activeNodeProps.rotationSpeedY, currentTargetConfig.targetRotationSpeed.y, lerpFactor);

                const mat = mainDodecahedron.material;
                mat.emissive.copy(activeNodeProps.emissive);
                mat.emissiveIntensity = activeNodeProps.emissiveIntensity;
                mainDodecahedron.scale.set(activeNodeProps.scale, activeNodeProps.scale, activeNodeProps.scale);
            }
        } else { // Fully faded out
            if (mainDodecahedron.visible) mainDodecahedron.visible = false;
            if (mainParticleSystem && mainParticleSystem.visible) mainParticleSystem.visible = false;
        }

        // --- Handle Specific Section Animations ---
        if (currentTargetConfig.animationType === 'unifiedIntelligence') animateUnifiedIntelligence(time, deltaTime);
        else if (currentTargetConfig.animationType === 'solutionsPipeline') animateSolutionsPipeline(time, deltaTime);
        else handleMainDodecaAnimation(currentTargetConfig, time, deltaTime);


        const currentTargetYPos = typeof targetNodeY.value === 'number' ? targetNodeY.value : 0;
        const groupYPosition = lerp(mainDodecahedron.position.y, currentTargetYPos, Math.min(deltaTime * 4.2, 1));

        mainDodecahedron.position.y = groupYPosition;
        unifiedIntelGroup.position.y = groupYPosition;
        solutionsPipelineGroup.position.y = groupYPosition;

        renderer.render(scene, camera);
    }

    try { init(); }
    catch (error) { /* ... (same) ... */
        console.error("EIDO Sentinel 3D Scene Initialization Error:", error);
        if (canvasContainer) canvasContainer.innerHTML = `<p style='color: #FF7043; background: #263238; padding: 15px; border-radius: 4px; font-family: monospace; text-align: left;'><b>3D Scene Initialization Failed.</b><br><span style='font-size:0.9em; color: #CFD8DC;'>${error.message}<br>Check console for details. (Often missing particle_glow.png or other assets)</span></p>`;
    }
});