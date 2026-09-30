let currentDataset = '2D_SRBA_OOOO';
let chartInstance = null;

const loadPage = () => {

    const statNumbers = document.querySelectorAll('.stat-number');
            
    const animateStats = () => {
        statNumbers.forEach(stat => {
            const target = +stat.getAttribute('data-target');
            const suffix = stat.getAttribute('data-suffix') || '';
            const duration = 1800; // Animation duration in ms
            const frameDuration = 1000 / 60;
            const totalFrames = Math.round(duration / frameDuration);
            let frame = 0;

            const counter = setInterval(() => {
                frame++;
                // Ease out quadratic
                const progress = frame / totalFrames;
                const current = Math.round(target * (1 - Math.pow(1 - progress, 3)));
                
                stat.innerText = current.toLocaleString() + suffix;

                if (frame === totalFrames) {
                    clearInterval(counter);
                    stat.innerText = target.toLocaleString() + suffix;
                }
            }, frameDuration);
        });
    };

    // Trigger animation when stats section is scrolled into view
    let observer = new IntersectionObserver((entries, observer) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                animateStats();
                observer.unobserve(entry.target);
            }
        });
    }, { threshold: 0.3 });

    const statsSection = document.querySelector('.stats-section');
    if (statsSection) {
        observer.observe(statsSection);
    }

    // -------------------------------------------------------------
    // BUILD MODEL TOGGLE PILLS
    // -------------------------------------------------------------
    const modelPillsContainer = document.getElementById('modelPills');
    const updateModelPills = () => {
        modelPillsContainer.innerHTML = ''; // Clear existing pills

        let datasetData = datasetsData[currentDataset];
        if (!datasetData) {
            console.log(`Current dataset not found.`);
            return;
        }

        models.forEach(m => {

            if (Object.keys(datasetData.scores).length > 0 && !datasetData.scores[m.id]) {
                // If the model has no scores for the current dataset, skip rendering its pill
                return;
            }

            const label = document.createElement('label');
            label.className = `model-pill-label ${m.visible ? 'active' : ''}`;
            label.style.borderColor = m.visible ? m.color : 'var(--glass-border)';
            
            label.innerHTML = `
                <input type="checkbox" value="${m.id}" ${m.visible ? 'checked' : ''}>
                <span class="dot" style="background-color: ${m.color}"></span>
                <div class="model-info">
                    <div class="model-name" title="Model Name">
                        ${m.name}
                        <span class="info" title="Sequence Info">${m.info}</span>
                    </div>
                    ${m.fineTune ? `<span class="fineTune" title="Fine-tuned">Finetuned</span>` : ''}
                </div>
            `;

            label.querySelector('input').addEventListener('change', (e) => {
                m.visible = e.target.checked;
                label.classList.toggle('active', m.visible);
                label.style.borderColor = m.visible ? m.color : 'var(--glass-border)';
                updateChart();
            });

            modelPillsContainer.appendChild(label);
        });
    };
    updateModelPills();

    // -------------------------------------------------------------
    // MULTI-SELECT STRATEGY TOGGLE LOGIC
    // -------------------------------------------------------------
    const strategyCards = document.querySelectorAll('#strategyOptions .strategy-card');
    strategyCards.forEach(card => {
        card.addEventListener('click', () => {
            const stratId = card.getAttribute('data-strat');
            const checkbox = card.querySelector('input[type="checkbox"]');
            
            checkbox.checked = !checkbox.checked;
            card.classList.toggle('active', checkbox.checked);

            // Update strategy internal visibility state
            const targetStrat = strategies.find(s => s.id === stratId);
            if (targetStrat) targetStrat.visible = checkbox.checked;

            updateChart();
        });
    });

    // -------------------------------------------------------------
    // INITIALIZE CHART.JS RADAR
    // -------------------------------------------------------------
    const ctx = document.getElementById('neptunaRadarCanvas').getContext('2d');

    // 1. Custom Plugin definition
    const diagonalTicksPlugin = {
        id: 'diagonalTicks',
        afterDraw(chart) {
            const { ctx, scales: { r } } = chart;
            const centerX = r.xCenter;
            const centerY = r.yCenter;
            const angleInRadians = (45 - 90) * (Math.PI / 180); // 45 degrees in standard unit circle

            const tickOpts = chart.options.scales.r.ticks;
            const textColor = tickOpts.color || '#00f2fe';
            const backdropColor = tickOpts.backdropColor || 'rgba(6, 10, 18, 0.8)';

            ctx.save();
            ctx.font = '600 10px "Plus Jakarta Sans"';
            ctx.fillStyle = '#919eab';
            ctx.textAlign = 'center';
            ctx.textBaseline = 'middle';

            // Loop over calculated tick values
            r.ticks.forEach((tick) => {
                const distance = r.getDistanceFromCenterForValue(tick.value * 0.85);
                const x = centerX + distance * Math.cos(angleInRadians);
                const y = centerY + distance * Math.sin(angleInRadians);

                // Draw background box (optional)
                const text = tick.value.toString();
                // display every second tick to avoid clutter
                if ([0.1, 0.3, 0.5, 0.7, 0.9].includes(tick.value)) return;
                const textWidth = ctx.measureText(text).width;
                ctx.fillStyle = backdropColor;
                ctx.fillRect(x - textWidth / 2 - 3, y - 6, textWidth + 6, 12);

                // Draw tick text
                ctx.fillStyle = textColor;
                ctx.fillText(text, x, y);
            });

            ctx.restore();
        }
    };

    function getChartColors(theme) {
        const isDark = theme === 'dark';
        
        return {
            angleLines: isDark ? 'rgba(255, 255, 255, 0.15)' : 'rgba(15, 23, 42, 0.12)',
            grid: isDark ? 'rgba(255, 255, 255, 0.08)' : 'rgba(15, 23, 42, 0.08)',
            pointLabels: isDark ? '#f0f6fc' : '#0b1324',
            ticksColor: isDark ? '#00f2fe' : '#008ca3',
            ticksBackdrop: isDark ? 'rgba(6, 10, 18, 0.85)' : 'rgba(244, 247, 252, 0.85)',
            legendText: isDark ? '#919eab' : '#5a6a85',
            tooltipBg: isDark ? 'rgba(6, 10, 18, 0.95)' : 'rgba(255, 255, 255, 0.95)',
            tooltipBorder: isDark ? 'rgba(0, 242, 254, 0.4)' : 'rgba(0, 140, 163, 0.35)',
            tooltipTitle: isDark ? '#ffffff' : '#0b1324',
            tooltipBody: isDark ? '#f0f6fc' : '#334155'
        };
    }

    function initChart() {
        const colors = getChartColors(document.documentElement.getAttribute('data-theme'));

        chartInstance = new Chart(ctx, {
            type: 'radar',
            data: {
                labels: radarMetrics,
                datasets: []
            },
            plugins: [diagonalTicksPlugin],
            options: {
                responsive: true,
                maintainAspectRatio: true,
                aspectRatio: 1.37,
                scales: {
                    r: {
                        angleLines: { color: colors.angleLines },
                        grid: { color: colors.grid },
                        pointLabels: {
                            color: colors.pointLabels,
                            font: { family: 'Plus Jakarta Sans', size: 11, weight: '600' }
                        },
                        ticks: {
                            display: false,
                            min: 0,
                            max: 1,
                            color: colors.ticksColor,
                            backdropColor: colors.ticksBackdrop,
                            backdropPadding: 4,
                            font: {
                                family: 'Plus Jakarta Sans',
                                size: 10,
                                weight: '600'
                            }
                        },
                        suggestedMin: 0,
                        suggestedMax: 1
                    }
                },
                plugins: {
                    legend: {
                        display: true,
                        position: 'bottom',
                        labels: {
                            color: colors.legendText,
                            font: { family: 'Plus Jakarta Sans', size: 11 },
                            padding: 14,
                            usePointStyle: true,
                            boxWidth: 8
                        }
                    },
                    tooltip: {
                        backgroundColor: colors.tooltipBg,
                        borderColor: colors.tooltipBorder,
                        borderWidth: 1,
                        titleColor: colors.tooltipTitle,
                        bodyColor: colors.tooltipBody,
                        titleFont: { family: 'Space Grotesk', size: 13 },
                        bodyFont: { family: 'Plus Jakarta Sans', size: 12 },
                        padding: 10,
                        displayColors: true
                    }
                }
            }
        });
        updateChart();
    }

    function updateChartTheme(theme) {
        if (!chartInstance) return;

        const colors = getChartColors(theme);
        const rScale = chartInstance.options.scales.r;

        rScale.angleLines.color = colors.angleLines;
        rScale.grid.color = colors.grid;
        rScale.pointLabels.color = colors.pointLabels;
        rScale.ticks.color = colors.ticksColor;
        rScale.ticks.backdropColor = colors.ticksBackdrop;

        chartInstance.options.plugins.legend.labels.color = colors.legendText;
        chartInstance.options.plugins.tooltip.backgroundColor = colors.tooltipBg;
        chartInstance.options.plugins.tooltip.borderColor = colors.tooltipBorder;
        chartInstance.options.plugins.tooltip.titleColor = colors.tooltipTitle;
        chartInstance.options.plugins.tooltip.bodyColor = colors.tooltipBody;

        chartInstance.update();
    }

    // -------------------------------------------------------------
    // UPDATE CHART DATA (GENERATE COMBINATIONS MODEL × STRATEGY)
    // -------------------------------------------------------------
    function updateChart() {
        if (!chartInstance) return;

        const dsData = datasetsData[currentDataset];
        const newDatasets = [];

        models.forEach(m => {
            if (m.visible && dsData.scores[m.id]) {
                
                // Helper for RGBA color conversion
                const hex = m.color;
                const r = parseInt(hex.slice(1, 3), 16);
                const g = parseInt(hex.slice(3, 5), 16);
                const b = parseInt(hex.slice(5, 7), 16);

                strategies.forEach(s => {
                    if (s.visible && dsData.scores[m.id][s.id]) {
                        const rawValues = dsData.scores[m.id][s.id];

                        newDatasets.push({
                            label: `${m.name} [${s.name}]`,
                            data: rawValues,
                            borderColor: m.color,
                            borderDash: s.dash,
                            backgroundColor: `rgba(${r}, ${g}, ${b}, ${s.fillAlpha})`,
                            borderWidth: 0.5,
                            pointBackgroundColor: m.color,
                            pointBorderColor: '#060a12',
                            pointHoverBackgroundColor: '#ffffff',
                            pointRadius: 3,
                            pointHoverRadius: 6
                        });
                    }
                });

            }
        });

        chartInstance.data.datasets = newDatasets;
        chartInstance.update();
    }

    // -------------------------------------------------------------
    // DATASET TAB SWITCHING LOGIC
    // -------------------------------------------------------------
    const tabButtons = document.querySelectorAll('#datasetTabs .tab-btn');
    tabButtons.forEach(btn => {
        btn.addEventListener('click', () => {
            tabButtons.forEach(b => b.classList.remove('active'));
            btn.classList.add('active');

            currentDataset = btn.getAttribute('data-dataset');
            const data = datasetsData[currentDataset];

            // Update Dataset Details
            document.getElementById('dsTitle').textContent = data.title;
            document.getElementById('dsDesc').textContent = data.desc;
            for (let i = 1; i <= 2; i++) {
                document.getElementById(`dsResolution${i}`).style.display = 'none';
            }
            let tmp = 0
            for (resolution of data.resolution) {
                tmp += 1
                document.getElementById(`dsResolution${tmp}`).innerHTML = `<img src="./assets/icons/border-all-solid-full.svg" alt="Border All Icon" class="section-icon"> ${resolution}`;
                document.getElementById(`dsResolution${tmp}`).style.display = 'flex';
            }
            document.getElementById('dsTrajectories').innerHTML = `<img src="./assets/icons/film-solid-full.svg" alt="Film Icon" class="section-icon"> ${data.trajectories}`;
            document.getElementById('dsPhysics').innerHTML = `<img src="./assets/icons/wind-solid-full.svg" alt="Wind Icon" class="section-icon"> ${data.physics}`;
            document.getElementById('dsDimensions').innerHTML = `<img src="./assets/icons/cube-solid-full.svg" alt="Cube Icon" class="section-icon"> ${data.dimensions || 'N/A'}`;

            // Update Video Source
            const video = document.getElementById('dsVideo');
            const videoSrc = document.getElementById('dsVideoSrc');
            videoSrc.src = data.video;
            video.load();
            video.play().catch(() => {});

            // Refresh Radar Chart
            updateChart();
            updateModelPills(); // Refresh model pills to reflect any dataset-specific visibility changes
        });
    });
    tabButtons[0].click(); // Activate the first tab by default

    document.getElementById('copyCitationBtn')?.addEventListener('click', async function() {
        const citationText = document.getElementById('citationText').innerText;
        const button = this;
        const label = button.querySelector('span');

        try {
            await navigator.clipboard.writeText(citationText);
            
            // Success Feedback
            button.classList.add('copied');
            label.textContent = 'Copied!';

            // Reset after 2 seconds
            setTimeout(() => {
                button.classList.remove('copied');
                label.textContent = 'Copy';
            }, 2000);
        } catch (err) {
            console.error('Failed to copy text: ', err);
        }
    });

    // Run chart initialization
    initChart();

    const toggleBtn = document.getElementById('theme-toggle');
    const STORAGE_KEY = 'user-theme';

    // 1. Determine initial theme: OS preference > Light default
    const getPreferredTheme = () => {
        return window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';
    };

    // 2. Apply theme state to root element & aria label
    const setTheme = (theme) => {
        updateChartTheme(theme)
        document.documentElement.setAttribute('data-theme', theme);
        
        if (toggleBtn) {
        toggleBtn.setAttribute(
            'aria-label',
            theme === 'dark' ? 'Switch to light mode' : 'Switch to dark mode'
        );
        }
    };

    // Initialize immediately to prevent flash of unstyled content
    setTheme(getPreferredTheme());

    // 3. Toggle listener
    if (toggleBtn) {
        toggleBtn.addEventListener('click', () => {
        const currentTheme = document.documentElement.getAttribute('data-theme');
        const nextTheme = currentTheme === 'dark' ? 'light' : 'dark';
        setTheme(nextTheme);
        });
    }

    // 4. Update automatically if OS theme changes and user hasn't explicitly set a preference
    window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', (e) => {
        setTheme(e.matches ? 'dark' : 'light');
    });

    const titleElement = document.getElementById('main-title'); // target your title
    const revealHeader = document.getElementById('reveal-header');

    if (!titleElement || !revealHeader) return;

    const observerOptions = {
        root: null,
        threshold: 0 // Trigger as soon as the title completely leaves or enters
    };

    const observer2 = new IntersectionObserver(([entry]) => {
        console.log("hi")
        // Reveal header only if the title is no longer visible AND has scrolled past the top
        const isPastTitle = !entry.isIntersecting && entry.boundingClientRect.top < 0;

        if (isPastTitle) {
            revealHeader.classList.add('is-visible');
        } else {
            revealHeader.classList.remove('is-visible');
        }
    }, observerOptions);

    observer2.observe(titleElement);

    window.addEventListener('resize', () => {
        document.getElementById("neptunaRadarCanvas").style.width = "100%";
        if (chartInstance) {
            chartInstance.resize();
        }
    });
}