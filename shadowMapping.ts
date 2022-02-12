import { mat4, vec3 } from "gl-matrix";
import dragonRawData from "stanford-dragon/4";

// Compute surface normals
// mesh.normals = computeSurfaceNormals(mesh.positions, mesh.triangles)

// Compute some easy uvs for testing

function computeSurfaceNormals(
  positions: [number, number, number][],
  triangles: [number, number, number][]
): [number, number, number][] {
  const normals: [number, number, number][] = positions.map(() => [0, 0, 0]);
  triangles.forEach(([i0, i1, i2]) => {
    const p0 = positions[i0];
    const p1 = positions[i1];
    const p2 = positions[i2];
    const v0 = vec3.subtract(vec3.create(), p1, p0);
    const v1 = vec3.subtract(vec3.create(), p2, p0);
    vec3.normalize(v0, v0);
    vec3.normalize(v1, v1);
    const norm = vec3.cross(vec3.create(), v0, v1);
    // Acuumulate the normals.
    vec3.add(normals[i0], normals[i0], norm);
    vec3.add(normals[i1], normals[i1], norm);
    vec3.add(normals[i2], normals[i2], norm);
  });
  normals.forEach((n) => {
    // Normalize accumulated normals.
    vec3.normalize(n, n);
  });

  return normals;
}

type ProjectedPlane = "xy" | "xz" | "yz";

const projectedPlane2Ids: { [a in ProjectedPlane]: [number, number] } = {
  xy: [0, 1],
  xz: [0, 2],
  yz: [1, 2],
};

function computeProjectedPlaneUVs(
  positions: [number, number, number][],
  projectedPlane: ProjectedPlane = "xy"
) {
  const idxs = projectedPlane2Ids[projectedPlane];
  const uvs: [number, number][] = positions.map(() => {
    // Initialize to zero
    return [0, 0];
  });
  const extentMin = [Infinity, Infinity]; // 各座標の内、UV座標が最も小さいもの
  const extentMax = [-Infinity, -Infinity];
  positions.forEach((pos, i) => {
    uvs[i][0] = pos[idxs[0]];
    uvs[i][1] = pos[idxs[1]];

    extentMin[0] = Math.min(pos[idxs[0]], extentMin[0]);
    extentMin[1] = Math.min(pos[idxs[1]], extentMin[1]);
    extentMax[0] = Math.max(pos[idxs[0]], extentMax[0]);
    extentMax[1] = Math.max(pos[idxs[1]], extentMax[1]);
  });
  uvs.forEach((uv) => {
    uv[0] = (uv[0] - extentMin[0]) / (extentMax[0] - extentMin[0]);
    uv[1] = (uv[1] - extentMin[1]) / (extentMax[1] - extentMin[1]);
  });
  return uvs;
}

const mesh = {
  positions: dragonRawData.positions as [number, number, number][],
  triangles: dragonRawData.cells as [number, number, number][],
  normals: [] as [number, number, number][],
  uvs: [] as [number, number][],
};

mesh.normals = computeSurfaceNormals(mesh.positions, mesh.triangles);
mesh.uvs = computeProjectedPlaneUVs(mesh.positions, "xy");

mesh.triangles.push(
  [mesh.positions.length, mesh.positions.length + 2, mesh.positions.length + 1],
  [mesh.positions.length, mesh.positions.length + 1, mesh.positions.length + 3]
);

// Push vertex attributes for an additional ground plane
mesh.positions.push(
  [-100, 20, -100],
  [100, 20, 100],
  [-100, 20, 100],
  [100, 20, -100]
);

mesh.normals.push([0, 1, 0], [0, 1, 0], [0, 1, 0], [0, 1, 0]);

mesh.uvs.push([0, 0], [1, 1], [0, 1], [1, 0]);

export async function init() {
  const adapter = await navigator.gpu.requestAdapter();
  const device = await adapter.requestDevice();
  const canvas = document.querySelector("canvas");
  const context = canvas.getContext("webgpu");
  const devicePixelRatio = window.devicePixelRatio || 1;
  const presentationSize = [
    canvas.clientWidth * devicePixelRatio,
    canvas.clientHeight * devicePixelRatio,
  ];
  const aspect = presentationSize[0] / presentationSize[1];
  const presentationFormat = context.getPreferredFormat(adapter);
  context.configure({
    device,
    format: presentationFormat,
    size: presentationSize,
  });

  // Create the model vertex buffer.
  const vertexBuffer = device.createBuffer({
    size: mesh.positions.length * 3 * 2 * Float32Array.BYTES_PER_ELEMENT,
    usage: GPUBufferUsage.VERTEX,
    mappedAtCreation: true,
  });
  {
    const mapping = new Float32Array(vertexBuffer.getMappedRange());
    for (let i = 0; i < mesh.positions.length; ++i) {
      mapping.set(mesh.positions[i], 6 * i);
      mapping.set(mesh.positions[i], 6 * i + 3);
    }
    vertexBuffer.unmap();
  }

  // Create the model index buffer.
  const indexCount = mesh.triangles.length * 3;
  const indexBuffer = device.createBuffer({
    size: indexCount * Uint16Array.BYTES_PER_ELEMENT,
    usage: GPUBufferUsage.INDEX,
    mappedAtCreation: true,
  });
  {
    const mapping = new Uint16Array(indexBuffer.getMappedRange());
    for (let i = 0; i < mesh.triangles.length; ++i) {
      mapping.set(mesh.triangles[i], 3 * i);
    }
    indexBuffer.unmap();
  }

  // Create the depth texture for rendering/sampling the shadow map.
  const shadowDepthTexture = device.createTexture({
    size: [1024, 1024, 1],
    usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
    format: "depth32float",
  });
  const shadowDepthTextureView = shadowDepthTexture.createView();

  // Create some common descriptors used for both the shadow pipeline
  // and the color rendering pipeline.
  const vertexBufferLayouts: Iterable<GPUVertexBufferLayout> = [
    {
      arrayStride: Float32Array.BYTES_PER_ELEMENT * 6,
      attributes: [
        {
          shaderLocation: 0,
          offset: 0,
          format: "float32x3",
        },
        {
          shaderLocation: 1,
          offset: Float32Array.BYTES_PER_ELEMENT * 3,
          format: "float32x3",
        },
      ],
    },
  ];

  const primitive: GPUPrimitiveState = {
    topology: "triangle-list",
    cullMode: "back",
  };

  const uniformBufferBindGroupLayout = device.createBindGroupLayout({
    entries: [
      {
        binding: 0,
        visibility: GPUShaderStage.VERTEX,
        buffer: {
          type: "uniform",
        },
      },
    ],
  });

  const shadowPipeline = device.createRenderPipeline({
    layout: device.createPipelineLayout({
      bindGroupLayouts: [
        uniformBufferBindGroupLayout,
        uniformBufferBindGroupLayout,
      ],
    }),
    vertex: {
      module: device.createShaderModule({
        code: `
        struct Scene {
            lightViewProjMatrix : mat4x4<f32>;
            cameraViewProjMatrix: mat4x4<f32>;
            lightPos : vec3<f32>;
        }

        struct Model {
            modelMatrix : mat4x4<f32>;
        }

        @group(0) @binding(0) var<uniform> scene : Scene;
        @group(1) @binding(0) var<uniform> model : Model;

        @stage(vertex)
        fn main(@location(0) position : vec3<f32>) -> @builtin(position) vec4<f32> {
            return scene.lightViewProjMatrix * model.modelMatrix * vec4<f32>(position, 1.0);
        }
        `,
      }),
      entryPoint: "main",
      buffers: vertexBufferLayouts,
    },
    depthStencil: {
      depthWriteEnabled: true,
      depthCompare: "less",
      format: "depth32float",
    },
    primitive,
  });

  // Create a bind group layout which holds the scene uniforms and the
  // texture+sampler for depth. We create it manually because the WebGPU
  // implementation doesn't infer this from the shader (yet)
  const bglForRender = device.createBindGroupLayout({
    entries: [
      {
        binding: 0,
        visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT,
        buffer: {
          type: "uniform",
        },
      },
      {
        binding: 1,
        visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT,
        texture: {
          sampleType: "depth",
        },
      },
      {
        binding: 2,
        visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT,
        sampler: {
          type: "comparison",
        },
      },
    ],
  });

  const shaderModule = device.createShaderModule({
    code: `
    struct Scene {
        lightViewProjMatrix : mat4x4<f32>;
        cameraViewProjMatrix: mat4x4<f32>;
        lightPos : vec3<f32>;
    }

    struct Model {
        modelMatrix : mat4x4<f32>;
    }

    @group(0) @binding(0) var<uniform> scene : Scene;
    @group(0) @binding(1) var shadowMap: texture_depth_2d;
    @group(0) @binding(2) var shadowSampler: sampler_comparison;
    @group(1) @binding(0) var<uniform> model : Model;

    struct VertexOutput {
        @location(0) shadowPos : vec3<f32>;
        @location(1) fragPos : vec3<f32>;
        @location(2) fragNorm : vec3<f32>;
        @builtin(position) Position : vec4<f32>;
    }
    

    @stage(vertex)
    fn vs_main(@location(0) position: vec3<f32>, @location(1) normal : vec3<f32>) -> VertexOutput {
        var output : VertexOutput;

        // XY is in (-1, 1) space, Z is in (0, 1) space
        let posFromLight : vec4<f32> = scene.lightViewProjMatrix * model.modelMatrix * vec4<f32>(position, 1.0);

        // Convert XY to (0, 1)
        // Y is flipped because texture coords are Y-down
        output.shadowPos = vec3<f32>(
            posFromLight.xy * vec2<f32>(0.5, -0.5) + vec2<f32>(0.5, 0.5),
            posFromLight.z
        );

        output.Position = scene.cameraViewProjMatrix * model.modelMatrix * vec4<f32>(position, 1.0);
        output.fragPos = output.Position.xyz;
        output.fragNorm = normal;

        return output;
    }

    let shadowDepthTextureSize : f32 = 1024.0;

    let albedo : vec3<f32> = vec3<f32>(0.9, 0.9, 0.9);
    let ambientFactor : f32 = 0.2;

    @stage(fragment)
    fn fs_main(input : VertexOutput) -> @location(0) vec4<f32> {
      // Percentage-closer filtering. Sample texels in the region
      // to smooth the result.
      var visibility : f32 = 0.0;
      let oneOverShadowDepthTextureSize = 1.0 / shadowDepthTextureSize;
      for (var y : i32 = -1 ; y <= 1 ; y = y + 1) {
          for (var x : i32 = -1 ; x <= 1 ; x = x + 1) {
            let offset : vec2<f32> = vec2<f32>(
              f32(x) * oneOverShadowDepthTextureSize,
              f32(y) * oneOverShadowDepthTextureSize);
    
              visibility = visibility + textureSampleCompare(
              shadowMap, shadowSampler,
              input.shadowPos.xy + offset, input.shadowPos.z - 0.007);
          }
      }
      visibility = visibility / 9.0;
    
      let lambertFactor : f32 = max(dot(normalize(scene.lightPos - input.fragPos), input.fragNorm), 0.0);
    
      let lightingFactor : f32 = min(ambientFactor + visibility * lambertFactor, 1.0);
      return vec4<f32>(lightingFactor * albedo, 1.0);
    }
      `,
  });

  const pipeline = device.createRenderPipeline({
    layout: device.createPipelineLayout({
      bindGroupLayouts: [bglForRender, uniformBufferBindGroupLayout],
    }),
    vertex: {
      module: shaderModule,
      entryPoint: "vs_main",
      buffers: vertexBufferLayouts,
    },
    fragment: {
      module: shaderModule,
      entryPoint: "fs_main",
      targets: [
        {
          format: presentationFormat,
        },
      ],
    },
    depthStencil: {
      depthWriteEnabled: true,
      depthCompare: "less",
      format: "depth24plus-stencil8",
    },
  });

  const depthTexture = device.createTexture({
    size: presentationSize,
    format: "depth24plus-stencil8",
    usage: GPUTextureUsage.RENDER_ATTACHMENT,
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

  const modelUniformBuffer = device.createBuffer({
    size: 4 * 16, // 4x4 matrix,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });

  const sceneUniformBuffer = device.createBuffer({
    size: 2 * 4 * 16 + 3 * 4,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });

  const sceneBindGroupForShadow = device.createBindGroup({
    layout: uniformBufferBindGroupLayout,
    entries: [
      {
        binding: 0,
        resource: {
          buffer: sceneUniformBuffer,
        },
      },
    ],
  });

  const sceneBindGroupForRender = device.createBindGroup({
    layout: bglForRender,
    entries: [
      {
        binding: 0,
        resource: {
          buffer: sceneUniformBuffer,
        },
      },
      {
        binding: 1,
        resource: shadowDepthTextureView,
      },
      {
        binding: 2,
        resource: device.createSampler({
          compare: "less",
        }),
      },
    ],
  });

  const modelBindGroup = device.createBindGroup({
    layout: uniformBufferBindGroupLayout,
    entries: [
      {
        binding: 0,
        resource: {
          buffer: modelUniformBuffer,
        },
      },
    ],
  });

  const eyePosition = vec3.fromValues(0, 50, -100);
  const upVector = vec3.fromValues(0, 1, 0);
  const origin = vec3.fromValues(0, 0, 0);

  const projectionMatrix = mat4.create();
  mat4.perspective(projectionMatrix, (2 * Math.PI) / 5, aspect, 1, 2000.0);
  const viewMatrix = mat4.create();
  mat4.lookAt(viewMatrix, eyePosition, origin, upVector);

  const lightPosition = vec3.fromValues(50, 100, -100);
  const lightViewMatrix = mat4.create();
  mat4.lookAt(lightViewMatrix, eyePosition, origin, upVector);

  const lightProjectionMatrix = mat4.create();
  {
    const left = -80;
    const right = 80;
    const bottom = -80;
    const top = 80;
    const near = -200;
    const far = 300;
    mat4.ortho(lightProjectionMatrix, left, right, bottom, top, near, far);
  }

  const lightViewProjMatrix = mat4.create();
  mat4.multiply(lightViewProjMatrix, lightProjectionMatrix, lightViewMatrix);

  const viewProjMatrix = mat4.create();
  mat4.multiply(viewProjMatrix, projectionMatrix, viewMatrix);

  // Move the model so it's centered.
  const modelMatrix = mat4.create();
  mat4.translate(modelMatrix, modelMatrix, vec3.fromValues(0, -5, 0));
  mat4.translate(modelMatrix, modelMatrix, vec3.fromValues(0, -40, 0));

  // The camera/light aren't moving, so write them into buffers now.
  {
    const lightMatrixData = lightViewProjMatrix as Float32Array;
    device.queue.writeBuffer(
      sceneUniformBuffer,
      0,
      lightMatrixData.buffer,
      lightMatrixData.byteOffset,
      lightMatrixData.byteLength
    );

    const cameraMatrixData = viewProjMatrix as Float32Array;
    device.queue.writeBuffer(
      sceneUniformBuffer,
      64,
      cameraMatrixData.buffer,
      cameraMatrixData.byteOffset,
      cameraMatrixData.byteLength
    );

    const lightData = lightPosition as Float32Array;
    device.queue.writeBuffer(
      sceneUniformBuffer,
      128,
      lightData.buffer,
      lightData.byteOffset,
      lightData.byteLength
    );

    const modelData = modelMatrix as Float32Array;
    device.queue.writeBuffer(
      modelUniformBuffer,
      0,
      modelData.buffer,
      modelData.byteOffset,
      modelData.byteLength
    );
  }

  function getCameraViewProjMatrix() {
    const eyePosition = vec3.fromValues(0, 50, -100);
    const rad = Math.PI * (Date.now() / 2000);
    vec3.rotateY(eyePosition, eyePosition, origin, rad);

    const viewMatrix = mat4.create();
    mat4.lookAt(viewMatrix, eyePosition, origin, upVector);

    mat4.multiply(viewProjMatrix, projectionMatrix, viewMatrix);
    return viewProjMatrix as Float32Array;
  }

  const shadowPassDescriptor: GPURenderPassDescriptor = {
    colorAttachments: [],
    depthStencilAttachment: {
      view: shadowDepthTextureView,
      depthLoadValue: 1.0,
      depthStoreOp: "store",
      stencilLoadValue: 0,
      stencilStoreOp: "store",
    },
  };

  function frame() {
    if (!canvas) return;
    const cameraViewProj = getCameraViewProjMatrix();
    device.queue.writeBuffer(
      sceneUniformBuffer,
      64,
      cameraViewProj.buffer,
      cameraViewProj.byteOffset,
      cameraViewProj.byteLength
    );

    renderPassDescriptor.colorAttachments[0].view = context
      .getCurrentTexture()
      .createView();

    const commandEncoder = device.createCommandEncoder();
    {
      const shadowPass = commandEncoder.beginRenderPass(shadowPassDescriptor);
      shadowPass.setPipeline(shadowPipeline);
      shadowPass.setBindGroup(0, sceneBindGroupForShadow);
      shadowPass.setBindGroup(1, modelBindGroup);
      shadowPass.setVertexBuffer(0, vertexBuffer);
      shadowPass.setIndexBuffer(indexBuffer, "uint16");
      shadowPass.drawIndexed(indexCount);

      shadowPass.endPass();
    }
    {
      const renderPass = commandEncoder.beginRenderPass(renderPassDescriptor);
      renderPass.setPipeline(pipeline);
      renderPass.setBindGroup(0, sceneBindGroupForRender);
      renderPass.setBindGroup(1, modelBindGroup);
      renderPass.setVertexBuffer(0, vertexBuffer);
      renderPass.setIndexBuffer(indexBuffer, "uint16");
      renderPass.drawIndexed(indexCount);

      renderPass.endPass();
    }
    device.queue.submit([commandEncoder.finish()]);
    requestAnimationFrame(frame);
  }

  requestAnimationFrame(frame);
}
