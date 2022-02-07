import { mat4, vec3 } from "gl-matrix";

export async function init() {
    const adapter = await navigator.gpu.requestAdapter();
    const device = await adapter.requestDevice();
    const canvas = document.querySelector("canvas") as HTMLCanvasElement;
    const context = canvas.getContext("webgpu");
    const devicePixelRatio = window.devicePixelRatio || 1;
    const presentationSize = [
        canvas.clientWidth * devicePixelRatio,
        canvas.clientHeight * devicePixelRatio
    ]
    const presentationFormat = context.getPreferredFormat(adapter);
    context.configure({device, format: presentationFormat, size: presentationSize});

    const cubeVertexSize = 4 * 10;
    const cubePositionOffset = 0;
    const cubeColorOffset = 4 * 4;
    const cubeUVOffset = 4 * 8;
    const cubeVertexCount = 36;
    const cubeVertexArray = new Float32Array([
        // float4 position, float4 color, float2 uv,
        1, -1, 1, 1,   1, 0, 1, 1,  1, 1,
        -1, -1, 1, 1,  0, 0, 1, 1,  0, 1,
        -1, -1, -1, 1, 0, 0, 0, 1,  0, 0,
        1, -1, -1, 1,  1, 0, 0, 1,  1, 0,
        1, -1, 1, 1,   1, 0, 1, 1,  1, 1,
        -1, -1, -1, 1, 0, 0, 0, 1,  0, 0,
      
        1, 1, 1, 1,    1, 1, 1, 1,  1, 1,
        1, -1, 1, 1,   1, 0, 1, 1,  0, 1,
        1, -1, -1, 1,  1, 0, 0, 1,  0, 0,
        1, 1, -1, 1,   1, 1, 0, 1,  1, 0,
        1, 1, 1, 1,    1, 1, 1, 1,  1, 1,
        1, -1, -1, 1,  1, 0, 0, 1,  0, 0,
      
        -1, 1, 1, 1,   0, 1, 1, 1,  1, 1,
        1, 1, 1, 1,    1, 1, 1, 1,  0, 1,
        1, 1, -1, 1,   1, 1, 0, 1,  0, 0,
        -1, 1, -1, 1,  0, 1, 0, 1,  1, 0,
        -1, 1, 1, 1,   0, 1, 1, 1,  1, 1,
        1, 1, -1, 1,   1, 1, 0, 1,  0, 0,
      
        -1, -1, 1, 1,  0, 0, 1, 1,  1, 1,
        -1, 1, 1, 1,   0, 1, 1, 1,  0, 1,
        -1, 1, -1, 1,  0, 1, 0, 1,  0, 0,
        -1, -1, -1, 1, 0, 0, 0, 1,  1, 0,
        -1, -1, 1, 1,  0, 0, 1, 1,  1, 1,
        -1, 1, -1, 1,  0, 1, 0, 1,  0, 0,
      
        1, 1, 1, 1,    1, 1, 1, 1,  1, 1,
        -1, 1, 1, 1,   0, 1, 1, 1,  0, 1,
        -1, -1, 1, 1,  0, 0, 1, 1,  0, 0,
        -1, -1, 1, 1,  0, 0, 1, 1,  0, 0,
        1, -1, 1, 1,   1, 0, 1, 1,  1, 0,
        1, 1, 1, 1,    1, 1, 1, 1,  1, 1,
      
        1, -1, -1, 1,  1, 0, 0, 1,  1, 1,
        -1, -1, -1, 1, 0, 0, 0, 1,  0, 1,
        -1, 1, -1, 1,  0, 1, 0, 1,  0, 0,
        1, 1, -1, 1,   1, 1, 0, 1,  1, 0,
        1, -1, -1, 1,  1, 0, 0, 1,  1, 1,
        -1, 1, -1, 1,  0, 1, 0, 1,  0, 0,
      ]);

    const verticesBuffer = device.createBuffer({
        size: cubeVertexArray.byteLength,
        usage: GPUBufferUsage.VERTEX,
        mappedAtCreation: true
    });
    new Float32Array(verticesBuffer.getMappedRange()).set(cubeVertexArray);
    verticesBuffer.unmap();

    const vsModule = device.createShaderModule({
        code: `
        struct Uniforms {
            mvpMat : mat4x4<f32>;
        }

        @binding(0) @group(0) var<uniform> uniforms: Uniforms;

        struct VertexOutput {
            @builtin(position) pos: vec4<f32>;
            @location(0) fragUV: vec2<f32>;
            @location(1) fragPosition: vec4<f32>;
        }

        @stage(vertex)
        fn main(@location(0) pos : vec4<f32>, @location(1) uv: vec2<f32>) -> VertexOutput {
            var output: VertexOutput;
            output.pos = uniforms.mvpMat * pos;
            output.fragUV = uv;
            output.fragPosition = 0.5 * (pos * vec4<f32>(1.0, 1.0, 1.0, 1.0));
            return output;
        }
        `
    });

    const fsModule = device.createShaderModule({
        code: `
        struct VertexOutput {
            @builtin(position) pos: vec4<f32>;
            @location(0) fragUV: vec2<f32>;
            @location(1) fragPosition: vec4<f32>;
        }

        @stage(fragment)
        fn main(input: VertexOutput) -> @location(0) vec4<f32> {
            return input.fragPosition;
        }
        `
    })

    const pipeline = device.createRenderPipeline({
        vertex: {
            module: vsModule,
            entryPoint: "main",
            buffers: [
                {
                    arrayStride: cubeVertexSize,
                    attributes: [
                        {
                            shaderLocation: 0,
                            offset: cubePositionOffset,
                            format: "float32x4"
                        },
                        {
                            shaderLocation: 1,
                            offset: cubeUVOffset,
                            format: "float32x2"
                        }
                    ]
                }
            ]
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
            topology: "triangle-list",
            cullMode: "back"
        },
        depthStencil: {
            depthWriteEnabled: true,
            depthCompare: "less",
            format: "depth24plus"
        }
    });

    const depthTexture = device.createTexture({
        size: presentationSize,
        format: "depth24plus",
        usage: GPUTextureUsage.RENDER_ATTACHMENT
    });

    const matrixSize = 4 * 16;
    const offset = 256;
    const uniformBufferSize = offset + matrixSize;
    const uniformBuffer = device.createBuffer({
        size: uniformBufferSize,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
    });
    const uniformBindGroup1 = device.createBindGroup({
        layout: pipeline.getBindGroupLayout(0),
        entries: [
            {
                binding: 0,
                resource: {
                    buffer: uniformBuffer,
                    offset: 0,
                    size: matrixSize
                }
            }
        ]
    })

    const uniformBindGroup2 = device.createBindGroup({
        layout: pipeline.getBindGroupLayout(0),
        entries: [
            {
                binding: 0,
                resource: {
                    buffer: uniformBuffer,
                    offset: offset,
                    size: matrixSize
                }
            }
        ]
    });

    const renderPassDescriptor : GPURenderPassDescriptor = {
        colorAttachments: [
            {
                view: undefined,
                loadValue: [0.5,0.5,0.5,1.0],
                storeOp: "store"
            }
        ],
        depthStencilAttachment: {
            view: depthTexture.createView(),
            depthLoadValue: 1.0,
            depthStoreOp: "store",
            stencilLoadValue: 0,
            stencilStoreOp: "store"
        }
    };

    const aspect = presentationSize[0] / presentationSize[1];
    const projMat = mat4.create();
    mat4.perspective(projMat, (2 * Math.PI) / 5, aspect, 1, 100);

    const modelMatrix1 = mat4.create();
    mat4.translate(modelMatrix1, modelMatrix1, vec3.fromValues(-2, 0, 0));
    
    const modelMatrix2 = mat4.create();
    mat4.translate(modelMatrix2, modelMatrix2, vec3.fromValues(2, 0, 0));

    const mvpMat1 = mat4.create() as Float32Array;
    const mvpMat2 = mat4.create() as Float32Array;

    const viewMat = mat4.create();
    mat4.translate(viewMat, viewMat, vec3.fromValues(0, 0, -7));

    const tmpMat41 = mat4.create();
    const tmpMat42 = mat4.create();

    function updateTransformationMatrix() {
        const now = Date.now() / 1000;
        mat4.rotate(
            tmpMat41,
            modelMatrix1,
            1,
            vec3.fromValues(Math.sin(now), Math.cos(now), 0)
        );
        mat4.rotate(
            tmpMat42,
            modelMatrix2,
            1,
            vec3.fromValues(Math.cos(now), Math.sin(now), 0)
        );

        mat4.multiply(mvpMat1, viewMat, tmpMat41);
        mat4.multiply(mvpMat1, projMat, mvpMat1);
        mat4.multiply(mvpMat2, viewMat, tmpMat42);
        mat4.multiply(mvpMat2, projMat, mvpMat2);
    }
    
    function frame() {
        updateTransformationMatrix();
        device.queue.writeBuffer(
            uniformBuffer,
            0,
            mvpMat1.buffer,
            mvpMat1.byteOffset,
            mvpMat1.byteLength
        );
        device.queue.writeBuffer(
            uniformBuffer,
            offset,
            mvpMat2.buffer,
            mvpMat2.byteOffset,
            mvpMat2.byteLength
        );
        renderPassDescriptor.colorAttachments[0].view = context.getCurrentTexture().createView();
        const commandEncoder = device.createCommandEncoder();
        const passEncoder = commandEncoder.beginRenderPass(renderPassDescriptor);
        passEncoder.setPipeline(pipeline);
        passEncoder.setVertexBuffer(0, verticesBuffer);
        passEncoder.setBindGroup(0, uniformBindGroup1);
        passEncoder.draw(cubeVertexCount, 1, 0, 0);
        passEncoder.setBindGroup(0, uniformBindGroup2);
        passEncoder.draw(cubeVertexCount, 1, 0, 0);
        passEncoder.endPass();

        device.queue.submit([commandEncoder.finish()]);

        requestAnimationFrame(frame);
    }

    requestAnimationFrame(frame);
}