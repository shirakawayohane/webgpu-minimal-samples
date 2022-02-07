import { mat4 } from "gl-matrix"

export async function init() {
    const adapter = await navigator.gpu.requestAdapter();
    const device = await adapter.requestDevice();
    const canvas = document.querySelector("canvas") as HTMLCanvasElement;
    const context = canvas.getContext("webgpu");
    const devicePixelRatio = window.devicePixelRatio || 1;
    const presentationSize = [
        canvas.clientWidth * devicePixelRatio,
        canvas.clientHeight * devicePixelRatio
    ];
    const presentationFormat = context.getPreferredFormat(adapter);

    context.configure({
        device,
        format: presentationFormat,
        size: presentationSize
    });

    const vsModule = device.createShaderModule({
        code: `@stage(vertex)
        fn main(@builtin(vertex_index) vertIdx : u32) -> @builtin(position) vec4<f32> {
            var pos = array<vec2<f32>, 3>(
                vec2<f32>(0.0, 0.5),
                vec2<f32>(-0.5, -0.5),
                vec2<f32>(0.5, -0.5)
            );
        
            return vec4<f32>(pos[vertIdx], 0.0, 1.0);
        }`
    });

    const fsModule = device.createShaderModule({
        code: `@stage(fragment)
        fn main() -> @location(0) vec4<f32> {
            return vec4<f32>(1.0, 0.0, 0.0, 1.0); // red
        }`
    });

    const pipeline = device.createRenderPipeline({
        vertex: {
            module: vsModule,
            entryPoint: "main"
        },
        fragment: {
            module: fsModule,
            entryPoint: "main",
            targets: [
                {
                    format: presentationFormat,
                }
            ]
        },
        primitive: {
            topology: "triangle-list"
        }
    })

    function frame() {
        const commandEncoder = device.createCommandEncoder();
        const textureView = context.getCurrentTexture().createView();
        const renderPassDesc: GPURenderPassDescriptor = {
            colorAttachments: [
                {
                    view: textureView,
                    loadValue: {r: 0, g:0, b: 0, a: 0},
                    storeOp: "store"
                }
            ]
        };

        const passEncoder = commandEncoder.beginRenderPass(renderPassDesc);
        passEncoder.setPipeline(pipeline);
        passEncoder.draw(3, 1, 0, 0);
        passEncoder.endPass();
        device.queue.submit([commandEncoder.finish()]);
        requestAnimationFrame(frame);
    }

    requestAnimationFrame(frame);
}