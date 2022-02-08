import { mat4, vec3 } from "gl-matrix";
import {
  cubePositionOffset,
  cubeUVOffset,
  cubeVertexArray,
  cubeVertexCount,
  cubeVertexSize,
} from "./cube";

export async function init() {
  const adapter = await navigator.gpu.requestAdapter();
  const device = await adapter.requestDevice();
  const canvas = document.querySelector("canvas") as HTMLCanvasElement;
  const context = canvas.getContext("webgpu");
  const devicePixelRatio = window.devicePixelRatio || 1;
  const presentationSize = [
    canvas.clientWidth * devicePixelRatio,
    canvas.clientHeight * devicePixelRatio,
  ];
  const presentationFormat = context.getPreferredFormat(adapter);
  context.configure({
    device,
    format: presentationFormat,
    size: presentationSize,
  });

  const vertBuf = device.createBuffer({
    size: cubeVertexArray.byteLength,
    usage: GPUBufferUsage.VERTEX,
    mappedAtCreation: true,
  });
  new Float32Array(vertBuf.getMappedRange()).set(cubeVertexArray);
  vertBuf.unmap();

  const vsModule = device.createShaderModule({
    code: `
        struct Uniforms {
            mvpMatrix : mat4x4<f32>;
        };
        
        @binding(0) @group(0) var<uniform> uniforms: Uniforms;

        struct VertexOutput {
            @builtin(position) pos : vec4<f32>;
            @location(0) fragUV: vec2<f32>;
            @location(1) fragPos: vec4<f32>;
        };

        @stage(vertex)
        fn main(@location(0) position: vec4<f32>, @location(1) uv: vec2<f32>) -> VertexOutput {
            var output : VertexOutput;
            output.pos = uniforms.mvpMatrix * position;
            output.fragUV = uv;
            output.fragPos = 0.5 * (position + vec4<f32>(1.0, 1.0, 1.0, 1.0));
            return output;
        }
        `,
  });

  const fsModule = device.createShaderModule({
    code: `
        struct VertexOutput {
            @builtin(position) pos : vec4<f32>;
            @location(0) fragUV: vec2<f32>;
            @location(1) fragPos: vec4<f32>;
        };

        @stage(fragment)
        fn main(input: VertexOutput) -> @location(0) vec4<f32> {
            return input.fragPos;
        }
        `,
  });

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
              format: "float32x4",
            },
            {
              shaderLocation: 1,
              offset: cubeUVOffset,
              format: "float32x2",
            },
          ],
        },
      ],
    },
    fragment: {
      module: fsModule,
      entryPoint: "main",
      targets: [
        {
          format: presentationFormat,
        },
      ],
    },
    primitive: {
      topology: "triangle-list",
      cullMode: "back",
    },
    depthStencil: {
      depthWriteEnabled: true,
      depthCompare: "less",
      format: "depth24plus",
    },
  });

  const depthTexture = device.createTexture({
    size: presentationSize,
    format: "depth24plus",
    usage: GPUTextureUsage.RENDER_ATTACHMENT,
  });

  const uniformBufferSize = 4 * 16;
  const uniformBuffer = device.createBuffer({
    size: uniformBufferSize,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });

  const uniformBindGroup = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      {
        binding: 0,
        resource: {
          buffer: uniformBuffer,
        },
      },
    ],
  });

  const renderPassDescriptor: GPURenderPassDescriptor = {
    colorAttachments: [
      {
        view: undefined,
        loadValue: [0.5, 0.5, 0.5, 1.0],
        storeOp: "store",
      },
    ],
    depthStencilAttachment: {
      view: depthTexture.createView(),
      depthLoadValue: 1.0,
      depthStoreOp: "store",
      stencilLoadValue: 0,
      stencilStoreOp: "store",
    },
  };

  const aspect = canvas.width / canvas.height;
  const projMat = mat4.create();
  mat4.perspective(projMat, (2 * Math.PI) / 5, aspect, 1, 100.0);
  function getTransformationMatrix() {
    const viewMat = mat4.create();
    mat4.translate(viewMat, viewMat, vec3.fromValues(0, 0, -4));
    const now = Date.now() / 1000;
    mat4.rotate(
      viewMat,
      viewMat,
      1,
      vec3.fromValues(Math.sin(now), Math.cos(now), 0)
    );
    const mvpMat = mat4.create();
    mat4.multiply(mvpMat, projMat, viewMat);
    return mvpMat as Float32Array;
  }

  function frame() {
    const transformationMatrix = getTransformationMatrix();
    device.queue.writeBuffer(
      uniformBuffer,
      0,
      transformationMatrix.buffer,
      transformationMatrix.byteOffset,
      transformationMatrix.byteLength
    );
    renderPassDescriptor.colorAttachments[0].view = context
      .getCurrentTexture()
      .createView();

    const commandEncoder = device.createCommandEncoder();
    const passEncoder = commandEncoder.beginRenderPass(renderPassDescriptor);
    passEncoder.setPipeline(pipeline);
    passEncoder.setBindGroup(0, uniformBindGroup);
    passEncoder.setVertexBuffer(0, vertBuf);
    passEncoder.draw(cubeVertexCount, 1, 0, 0);
    passEncoder.endPass();
    device.queue.submit([commandEncoder.finish()]);
    requestAnimationFrame(frame);
  }

  requestAnimationFrame(frame);
}
