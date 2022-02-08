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
            mvpMatrices : array<mat4x4<f32>, 16>;
        };
        
        @binding(0) @group(0) var<uniform> uniforms: Uniforms;

        struct VertexOutput {
            @builtin(position) pos : vec4<f32>;
            @location(0) fragUV: vec2<f32>;
            @location(1) fragPos: vec4<f32>;
        };

        @stage(vertex)
        fn main(@builtin(instance_index) instanceIdx : u32,
                @location(0) position: vec4<f32>,
                @location(1) uv: vec2<f32>) -> VertexOutput {
            var output : VertexOutput;
            output.pos = uniforms.mvpMatrices[instanceIdx] * position;
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

  const xCount = 4;
  const yCount = 4;
  const numInstances = xCount * yCount;
  const matrixFloatCount = 16; // 4x4 matrix
  const matrixSize = 4 * matrixFloatCount;
  const uniformBufferSize = numInstances * matrixSize;

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

  const aspect = presentationSize[0] / presentationSize[1];
  const projectionMatrix = mat4.create();
  mat4.perspective(projectionMatrix, (2 * Math.PI) / 5, aspect, 1, 100.0);

  const modelMatrices = new Array(numInstances);
  const mvpMatricesData = new Float32Array(matrixFloatCount * numInstances);

  const step = 4.0;

  // Initialize the matrix data for every instance.
  let m = 0;
  for (let x = 0; x < xCount; x++) {
    for (let y = 0; y < yCount; y++) {
      modelMatrices[m] = mat4.create();
      mat4.translate(
        modelMatrices[m],
        modelMatrices[m],
        vec3.fromValues(
          step * (x - xCount / 2 + 0.5),
          step * (y - yCount / 2 + 0.5),
          0
        )
      );
      m++;
    }
  }

  const viewMatrix = mat4.create();
  mat4.translate(viewMatrix, viewMatrix, vec3.fromValues(0, 0, -12));

  const tmpMat4 = mat4.create();

  // Update the transformation matrix data for each instance.
  function updateTransformationMatrix() {
    const now = Date.now() / 1000;

    let m = 0,
      i = 0;
    for (let x = 0; x < xCount; x++) {
      for (let y = 0; y < yCount; y++) {
        mat4.rotate(
          tmpMat4,
          modelMatrices[i],
          1,
          vec3.fromValues(
            Math.sin((x + 0.5) * now),
            Math.cos((y + 0.5) * now),
            0
          )
        );

        mat4.multiply(tmpMat4, viewMatrix, tmpMat4);
        mat4.multiply(tmpMat4, projectionMatrix, tmpMat4);

        mvpMatricesData.set(tmpMat4, m);

        i++;
        m += matrixFloatCount;
      }
    }
  }

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

  function frame() {
    updateTransformationMatrix();
    device.queue.writeBuffer(
      uniformBuffer,
      0,
      mvpMatricesData.buffer,
      mvpMatricesData.byteOffset,
      mvpMatricesData.byteLength
    );
    renderPassDescriptor.colorAttachments[0].view = context
      .getCurrentTexture()
      .createView();

    const commandEncoder = device.createCommandEncoder();
    const passEncoder = commandEncoder.beginRenderPass(renderPassDescriptor);
    passEncoder.setPipeline(pipeline);
    passEncoder.setBindGroup(0, uniformBindGroup);
    passEncoder.setVertexBuffer(0, vertBuf);
    passEncoder.draw(cubeVertexCount, numInstances, 0, 0);
    passEncoder.endPass();
    device.queue.submit([commandEncoder.finish()]);
    requestAnimationFrame(frame);
  }

  requestAnimationFrame(frame);
}
