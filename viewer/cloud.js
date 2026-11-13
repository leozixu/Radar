(() => {
    "use strict";

    const POINT_WIDTH = 5;
    const POSITION_WIDTH = 3;
    const FLOAT_BYTES = Float32Array.BYTES_PER_ELEMENT;
    const DEFAULT_YAW = -0.55;
    const DEFAULT_PITCH = -0.60;
    const RING_RADII = [0.25, 0.5, 0.75, 1];
    const DISTANCE_FORMAT = new Intl.NumberFormat("zh-CN", { maximumSignificantDigits: 4 });
    const VERTEX_SHADER = `
        attribute vec3 a_position;
        attribute vec2 a_meta;
        uniform vec3 u_right;
        uniform vec3 u_up;
        uniform float u_verticalCenter;
        uniform vec2 u_scale;
        uniform vec2 u_circles;
        uniform vec3 u_visible;
        uniform float u_pixelRatio;
        uniform mediump vec4 u_lineColor;
        varying vec4 v_color;

        void main() {
            v_color = u_lineColor;
            if (u_lineColor.a == 0.0) {
                float circle = a_meta.x;
                float kind = a_meta.y;
                float visible = u_visible.x;
                float size = 2.4;
                v_color = vec4(0.471, 0.549, 0.643, 0.35);

                if (kind == 1.0) {
                    visible = u_visible.y;
                    size = 3.8;
                    v_color = vec4(0.333, 0.882, 0.729, 0.92);
                } else if (kind == 2.0) {
                    visible = u_visible.z;
                    size = 3.3;
                    v_color = vec4(0.957, 0.725, 0.424, 0.88);
                }

                gl_PointSize = size * u_pixelRatio;
                if (visible < 0.5 || circle < u_circles.x || circle > u_circles.y) {
                    gl_Position = vec4(2.0, 2.0, 2.0, 1.0);
                    return;
                }
            }

            vec2 screen = vec2(dot(u_right, a_position), dot(u_up, a_position) - u_verticalCenter);
            gl_Position = vec4(screen * u_scale, 0.0, 1.0);
        }
    `;

    const FRAGMENT_SHADER = `
        precision mediump float;
        uniform mediump vec4 u_lineColor;
        varying vec4 v_color;

        void main() {
            if (u_lineColor.a > 0.0) {
                gl_FragColor = v_color;
                return;
            }
            float edge = length(2.0 * gl_PointCoord - 1.0);
            float opacity = 1.0 - smoothstep(0.56, 1.0, edge);
            gl_FragColor = vec4(v_color.rgb, v_color.a * opacity);
        }
    `;

    function compile(gl, type, source) {
        const shader = gl.createShader(type);
        gl.shaderSource(shader, source);
        gl.compileShader(shader);
        if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
            const message = gl.getShaderInfoLog(shader);
            gl.deleteShader(shader);
            throw new Error(`无法初始化三维着色器。${message || ""}`);
        }
        return shader;
    }

    function initialize(gl) {
        const program = gl.createProgram();
        const vertex = compile(gl, gl.VERTEX_SHADER, VERTEX_SHADER);
        const fragment = compile(gl, gl.FRAGMENT_SHADER, FRAGMENT_SHADER);
        gl.attachShader(program, vertex);
        gl.attachShader(program, fragment);
        gl.linkProgram(program);
        gl.deleteShader(vertex);
        gl.deleteShader(fragment);
        if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
            const message = gl.getProgramInfoLog(program);
            gl.deleteProgram(program);
            throw new Error(`无法初始化三维视图。${message || ""}`);
        }

        const position = gl.getAttribLocation(program, "a_position");
        const meta = gl.getAttribLocation(program, "a_meta");
        gl.useProgram(program);
        gl.enableVertexAttribArray(position);
        gl.clearColor(0, 0, 0, 0);
        gl.enable(gl.BLEND);
        gl.blendFuncSeparate(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA, gl.ONE, gl.ONE_MINUS_SRC_ALPHA);

        return {
            pointBuffer: gl.createBuffer(),
            lineBuffer: gl.createBuffer(),
            position,
            meta,
            right: gl.getUniformLocation(program, "u_right"),
            up: gl.getUniformLocation(program, "u_up"),
            verticalCenter: gl.getUniformLocation(program, "u_verticalCenter"),
            scale: gl.getUniformLocation(program, "u_scale"),
            circles: gl.getUniformLocation(program, "u_circles"),
            visible: gl.getUniformLocation(program, "u_visible"),
            pixelRatio: gl.getUniformLocation(program, "u_pixelRatio"),
            lineColor: gl.getUniformLocation(program, "u_lineColor"),
        };
    }

    function drawReference(ctx, width, height, project, right, scale, world, altitudeTop) {
        ctx.clearRect(0, 0, width, height);
        const origin = project([0, 0, 0]);
        ctx.textBaseline = "middle";
        ctx.textAlign = "center";

        function label(text, x, y, color, heading = false) {
            if (x < -12 || y < -12 || x > width + 12 || y > height + 12) return;
            ctx.font = heading ? "600 13px system-ui, sans-serif" : "11px system-ui, sans-serif";
            const margin = ctx.measureText(text).width / 2 + 4;
            const labelX = Math.max(margin, Math.min(width - margin, x));
            const labelY = Math.max(9, Math.min(height - 9, y));
            ctx.strokeStyle = "rgba(12, 20, 31, 0.82)";
            ctx.lineWidth = 3;
            ctx.lineJoin = "round";
            ctx.strokeText(text, labelX, labelY);
            ctx.fillStyle = color;
            ctx.fillText(text, labelX, labelY);
        }

        function formatDistance(value) {
            return world.maxRangeKm === null
                ? DISTANCE_FORMAT.format(value)
                : `${DISTANCE_FORMAT.format(value * world.maxRangeKm)} km`;
        }

        for (const radius of RING_RADII) {
            ctx.beginPath();
            ctx.moveTo(...project([radius, 0, 0]));
            for (let step = 1; step <= 72; step += 1) {
                const angle = step * Math.PI / 36;
                ctx.lineTo(...project([radius * Math.cos(angle), 0, radius * Math.sin(angle)]));
            }
            ctx.closePath();
            ctx.lineWidth = 1;
            ctx.strokeStyle = radius === 1 ? "rgba(134, 174, 203, 0.45)" : "rgba(134, 174, 203, 0.25)";
            ctx.stroke();
        }

        for (const radius of RING_RADII) {
            if (scale < 175 && (radius === 0.25 || radius === 0.75)) continue;
            const [x, y] = project([-radius * right[0], 0, -radius * right[2]]);
            label(formatDistance(radius), x, y + 12, "#9fb7cb");
        }

        const axes = [
            { name: "横向 X", start: [-1, 0, 0], end: [1, 0, 0] },
            { name: "前向 Y", start: [0, 0, -1], end: [0, 0, 1] },
            { name: "高度 Z", start: [0, world.minHeight, 0], end: [0, altitudeTop, 0] },
        ];
        for (const { name, start, end: endpoint } of axes) {
            const end = project(endpoint);
            const dx = end[0] - origin[0];
            const dy = end[1] - origin[1];
            const length = Math.hypot(dx, dy);
            if (length < 22) continue;
            const ux = dx / length;
            const uy = dy / length;
            const arrowX = end[0] - ux * 7;
            const arrowY = end[1] - uy * 7;

            ctx.beginPath();
            ctx.moveTo(...project(start));
            ctx.lineTo(...end);
            ctx.moveTo(arrowX - uy * 3, arrowY + ux * 3);
            ctx.lineTo(...end);
            ctx.lineTo(arrowX + uy * 3, arrowY - ux * 3);
            ctx.lineWidth = 1;
            ctx.strokeStyle = "rgba(166, 201, 222, 0.56)";
            ctx.stroke();

            if (endpoint[1]) {
                const unitLength = length / altitudeTop;
                const interval = unitLength >= 132 ? 0.25 : 0.5;
                for (let level = -1; level <= 1; level += interval) {
                    if (!level || level < world.minHeight || level > altitudeTop || Math.abs(level) * unitLength < 22) {
                        continue;
                    }
                    const [x, y] = project([0, level, 0]);
                    ctx.beginPath();
                    ctx.moveTo(x - 4, y);
                    ctx.lineTo(x, y);
                    ctx.lineWidth = 1;
                    ctx.strokeStyle = "rgba(166, 201, 222, 0.56)";
                    ctx.stroke();
                    const tick = formatDistance(level);
                    ctx.font = "11px system-ui, sans-serif";
                    label(tick, x - ctx.measureText(tick).width / 2 - 8, y, "#9fb7cb");
                }
            }
            label(name, end[0] + ux * 26, end[1] + uy * 16, "#d7e4f0", true);
        }

        ctx.beginPath();
        ctx.arc(origin[0], origin[1], 7, 0, 2 * Math.PI);
        ctx.strokeStyle = "#92d3ff";
        ctx.lineWidth = 1.25;
        ctx.fillStyle = "#101c2b";
        ctx.fill();
        ctx.stroke();
        ctx.beginPath();
        ctx.arc(origin[0], origin[1], 2.5, 0, 2 * Math.PI);
        ctx.fillStyle = "#d9f2ff";
        ctx.fill();
        label("雷达", origin[0], origin[1] + 18, "#bce5ff", true);
    }

    window.createRadarCloud = function createRadarCloud(canvas, axesCanvas) {
        const gl = canvas.getContext("webgl", { antialias: false });
        if (!gl) throw new Error("浏览器无法启用 WebGL。请检查浏览器的图形加速设置。");
        const axes = axesCanvas.getContext("2d");
        if (!axes) throw new Error("浏览器无法绘制坐标轴。");

        let uniforms = initialize(gl);
        let points = new Float32Array();
        let upload = true;
        let solidLines = new Float32Array();
        let dashedLines = new Float32Array();
        let focused = false;
        let uploadLines = false;
        let frame = 0;
        let yaw = DEFAULT_YAW;
        let pitch = DEFAULT_PITCH;
        let zoom = 1;
        let world = { maxRangeKm: null, minHeight: 0, maxHeight: 1 };
        let drag;
        let minCircle = -1e12;
        let maxCircle = 1e12;
        let visible = [1, 1, 1];

        function schedule() {
            if (!frame) frame = requestAnimationFrame(draw);
        }

        function draw() {
            frame = 0;
            if (gl.isContextLost()) return;
            const { width, height } = canvas.getBoundingClientRect();
            if (!width || !height) return;

            const dpr = Math.min(window.devicePixelRatio || 1, 2);
            const pixelWidth = Math.max(1, Math.round(width * dpr));
            const pixelHeight = Math.max(1, Math.round(height * dpr));
            for (const surface of [canvas, axesCanvas]) {
                if (surface.width !== pixelWidth) surface.width = pixelWidth;
                if (surface.height !== pixelHeight) surface.height = pixelHeight;
            }
            axes.setTransform(pixelWidth / width, 0, 0, pixelHeight / height, 0, 0);
            gl.viewport(0, 0, pixelWidth, pixelHeight);
            gl.clear(gl.COLOR_BUFFER_BIT);

            const sine = Math.sin(yaw);
            const cosine = Math.cos(yaw);
            const tilt = Math.sin(pitch);
            const vertical = Math.cos(pitch);
            const right = [cosine, 0, sine];
            const up = [sine * tilt, vertical, -cosine * tilt];
            const altitudeTop = Math.max(world.maxHeight, 0.25);
            // Fit the ground circle and the visible slice of the unit sphere at the current pitch.
            const upperY = Math.min(altitudeTop, vertical);
            const lowerY = Math.max(world.minHeight, -vertical);
            const upper = vertical * upperY + Math.abs(tilt) * Math.sqrt(1 - upperY ** 2);
            const lower = vertical * lowerY - Math.abs(tilt) * Math.sqrt(1 - lowerY ** 2);
            const verticalCenter = (upper + lower) / 2;
            const scale = Math.min(Math.max(width - 76, width * 0.7) / 2,
                Math.max(height - 60, height * 0.7) / (upper - lower)) * zoom;
            const project = ([x, y, z]) => [
                width / 2 + scale * (right[0] * x + right[2] * z),
                height / 2 - scale * (up[0] * x + up[1] * y + up[2] * z - verticalCenter),
            ];

            if (upload) {
                gl.bindBuffer(gl.ARRAY_BUFFER, uniforms.pointBuffer);
                gl.bufferData(gl.ARRAY_BUFFER, points, gl.STATIC_DRAW);
                upload = false;
            }
            if (uploadLines) {
                gl.bindBuffer(gl.ARRAY_BUFFER, uniforms.lineBuffer);
                gl.bufferData(gl.ARRAY_BUFFER, solidLines.byteLength + dashedLines.byteLength, gl.STATIC_DRAW);
                if (solidLines.length) gl.bufferSubData(gl.ARRAY_BUFFER, 0, solidLines);
                if (dashedLines.length) gl.bufferSubData(gl.ARRAY_BUFFER, solidLines.byteLength, dashedLines);
                uploadLines = false;
            }

            gl.uniform3fv(uniforms.right, right);
            gl.uniform3fv(uniforms.up, up);
            gl.uniform1f(uniforms.verticalCenter, verticalCenter);
            gl.uniform2f(uniforms.scale, 2 * scale / width, 2 * scale / height);
            if (points.length) {
                gl.bindBuffer(gl.ARRAY_BUFFER, uniforms.pointBuffer);
                const stride = POINT_WIDTH * FLOAT_BYTES;
                gl.vertexAttribPointer(uniforms.position, POSITION_WIDTH, gl.FLOAT, false, stride, 0);
                gl.enableVertexAttribArray(uniforms.meta);
                gl.vertexAttribPointer(uniforms.meta, 2, gl.FLOAT, false, stride, POSITION_WIDTH * FLOAT_BYTES);
                gl.uniform4f(uniforms.lineColor, 0, 0, 0, 0);
                gl.uniform2f(uniforms.circles, minCircle, maxCircle);
                gl.uniform3fv(uniforms.visible, visible);
                gl.uniform1f(uniforms.pixelRatio, dpr);
                gl.drawArrays(gl.POINTS, 0, points.length / POINT_WIDTH);
            }
            if (solidLines.length || dashedLines.length) {
                gl.bindBuffer(gl.ARRAY_BUFFER, uniforms.lineBuffer);
                gl.vertexAttribPointer(uniforms.position, POSITION_WIDTH, gl.FLOAT, false, 0, 0);
                gl.disableVertexAttribArray(uniforms.meta);
                const [red, green] = focused ? [193 / 255, 242 / 255] : [130 / 255, 220 / 255];
                const solidVertexCount = solidLines.length / POSITION_WIDTH;
                if (solidLines.length) {
                    gl.uniform4f(uniforms.lineColor, red, green, 1, focused ? 0.98 : 0.45);
                    gl.drawArrays(gl.LINES, 0, solidVertexCount);
                }
                if (dashedLines.length) {
                    gl.uniform4f(uniforms.lineColor, red, green, 1, focused ? 0.72 : 0.28);
                    gl.drawArrays(gl.LINES, solidVertexCount, dashedLines.length / POSITION_WIDTH);
                }
            }
            drawReference(axes, width, height, project, right, scale, world, altitudeTop);
        }

        function orbit(dx, dy) {
            yaw += dx;
            pitch = Math.max(-Math.PI / 2, Math.min(Math.PI / 2, pitch + dy));
            schedule();
        }

        function changeZoom(factor) {
            zoom = Math.max(0.43, Math.min(5, zoom * factor));
            schedule();
        }

        function resetView() {
            yaw = DEFAULT_YAW;
            pitch = DEFAULT_PITCH;
            zoom = 1;
            schedule();
        }

        function setTopView() {
            yaw = 0;
            pitch = -Math.PI / 2;
            zoom = 1;
            schedule();
        }

        canvas.addEventListener("pointerdown", event => {
            if (drag || event.button !== 0) return;
            drag = { id: event.pointerId, x: event.clientX, y: event.clientY };
            canvas.setPointerCapture(event.pointerId);
        });
        canvas.addEventListener("pointermove", event => {
            if (drag?.id !== event.pointerId) return;
            orbit((event.clientX - drag.x) * 0.008, (event.clientY - drag.y) * 0.008);
            drag.x = event.clientX;
            drag.y = event.clientY;
        });
        canvas.addEventListener("lostpointercapture", event => {
            if (drag?.id === event.pointerId) drag = undefined;
        });
        canvas.addEventListener("wheel", event => {
            event.preventDefault();
            const unit = event.deltaMode === 1 ? 16 : event.deltaMode === 2 ? 360 : 1;
            const delta = Math.max(-400, Math.min(400, event.deltaY * unit));
            changeZoom(Math.exp(-delta * 0.0016));
        }, { passive: false });
        canvas.addEventListener("keydown", event => {
            switch (event.key) {
                case "ArrowLeft": orbit(-0.12, 0); break;
                case "ArrowRight": orbit(0.12, 0); break;
                case "ArrowUp": orbit(0, -0.12); break;
                case "ArrowDown": orbit(0, 0.12); break;
                case "+": case "=": changeZoom(1.14); break;
                case "-": changeZoom(1 / 1.14); break;
                case "Home": resetView(); break;
                default: return;
            }
            event.preventDefault();
        });
        canvas.addEventListener("webglcontextlost", event => event.preventDefault());
        canvas.addEventListener("webglcontextrestored", () => {
            uniforms = initialize(gl);
            upload = true;
            uploadLines = true;
            schedule();
        });
        new ResizeObserver(schedule).observe(canvas);
        schedule();

        return {
            setData(data) {
                points = data;
                upload = true;
                schedule();
            },
            setFilter(filter) {
                ({ minCircle, maxCircle } = filter);
                visible = [Number(filter.showClutter), Number(filter.showTarget), Number(filter.showUnknown)];
                schedule();
            },
            setTrajectories(solid, dashed, focus = false) {
                solidLines = solid;
                dashedLines = dashed;
                focused = focus;
                uploadLines = true;
                schedule();
            },
            setWorld(settings) {
                world = settings;
                schedule();
            },
            setTopView,
            resetView,
        };
    };
})();
