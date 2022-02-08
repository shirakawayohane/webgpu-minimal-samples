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
  const canvas = document.querySelector("canvas");
  const context = canvas.getContext("webgpu");
  const devicePixelRatio = window.devicePixelRatio || 1;
  const presentationSize = [canvas.clientWidth, canvas.clientHeight];
  const presentationFormat = context.getPreferredFormat(adapter);
  context.configure({
    device,
    format: presentationFormat,
    usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC,
    size: presentationSize,
  });

  const verticesBuffer = device.createBuffer({
    size: cubeVertexArray.byteLength,
    usage: GPUBufferUsage.VERTEX,
    mappedAtCreation: true,
  });
  new Float32Array(verticesBuffer.getMappedRange()).set(cubeVertexArray);
  verticesBuffer.unmap();

  const vsModule = device.createShaderModule({
    code: `
struct Uniforms {
    mvpMat: mat4x4<f32>;
}

@binding(0) @group(0) var<uniform> uniforms: Uniforms;

struct VertexOutput {
    @builtin(position) Position: vec4<f32>;
    @location(0) fragUV : vec2<f32>;
    @location(1) fragPosition : vec4<f32>;
}

@stage(vertex)
fn main(@location(0) position: vec4<f32>, @location(1) uv : vec2<f32>) -> VertexOutput {
    var output : VertexOutput;
    output.Position = uniforms.mvpMat * position;
    output.fragUV = uv;
    output.fragPosition = 0.5 * (position + vec4<f32>(1.0, 1.0, 1.0, 1.0));
    return output;
}
      `,
  });

  const fsModule = device.createShaderModule({
    code: `
struct VertexOutput {
    @builtin(position) Position: vec4<f32>;
    @location(0) fragUV : vec2<f32>;
    @location(1) fragPosition : vec4<f32>;
}

@binding(1) @group(0) var mySampler : sampler;
@binding(2) @group(0) var myTexture : texture_2d<f32>;

@stage(fragment)
fn main(input : VertexOutput) -> @location(0) vec4<f32> {
    let texColor = textureSample(myTexture, mySampler, input.fragUV * 0.8 + vec2<f32>(0.1, 0.1));
    var f : f32;
    if (length(texColor.rgb - vec3<f32>(0.5, 0.5, 0.5)) < 0.01) {
        f = 1.0;
    } else {
        f = 0.0;
    }
    return (1.0 - f) * texColor + f * input.fragPosition;
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

  const uniformBufferSize = 4 * 16; // 4x4 matrix
  const uniformBuffer = device.createBuffer({
    size: uniformBufferSize,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.UNIFORM,
  });

  const cubeTexture = device.createTexture({
    size: presentationSize,
    format: presentationFormat,
    usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
  });

  const sampler = device.createSampler({
    magFilter: "linear",
    minFilter: "linear",
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
      {
        binding: 1,
        resource: sampler,
      },
      {
        binding: 2,
        resource: cubeTexture.createView(),
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

  const aspect = presentationSize[0] / presentationSize[1];
  const projectionMatrix = mat4.create();
  mat4.perspective(projectionMatrix, (2 * Math.PI) / 5, aspect, 1, 100);
  function getTransformationMatrix() {
    const viewMatrix = mat4.create();
    mat4.translate(viewMatrix, viewMatrix, vec3.fromValues(0, 0, -4));
    const now = Date.now() / 1000;
    mat4.rotate(
      viewMatrix,
      viewMatrix,
      1,
      vec3.fromValues(Math.sin(now), Math.cos(now), 0)
    );

    const mvpMatrix = mat4.create();
    mat4.multiply(mvpMatrix, projectionMatrix, viewMatrix);
    return mvpMatrix as Float32Array;
  }

  function frame() {
    const transformationMatrix = getTransformationMatrix();
    device.queue.writeTexture;
    device.queue.writeBuffer(
      uniformBuffer,
      0,
      transformationMatrix.buffer,
      transformationMatrix.byteOffset,
      transformationMatrix.byteLength
    );
    const swapChainTexture = context.getCurrentTexture();
    renderPassDescriptor.colorAttachments[0].view =
      swapChainTexture.createView();

    const commandEncoder = device.createCommandEncoder();
    const passEncoder = commandEncoder.beginRenderPass(renderPassDescriptor);
    passEncoder.setPipeline(pipeline);
    passEncoder.setBindGroup(0, uniformBindGroup);
    passEncoder.setVertexBuffer(0, verticesBuffer);
    passEncoder.draw(cubeVertexCount, 1, 0, 0);
    passEncoder.endPass();

    commandEncoder.copyTextureToTexture(
      {
        texture: swapChainTexture,
      },
      {
        texture: cubeTexture,
      },
      presentationSize
    );

    device.queue.submit([commandEncoder.finish()]);

    requestAnimationFrame(frame);
  }

  requestAnimationFrame(frame);
}
