export async function init() {
  const canvas = document.querySelector("canvas");
  const context = canvas.getContext("webgpu");
  const adapter = await navigator.gpu.requestAdapter();
  const device = await adapter.requestDevice();

  const devicePixelRatio = window.devicePixelRatio || 1;
  const presentationSize = [canvas.width, canvas.height]; // TODO
  const presentationFormat = context.getPreferredFormat(adapter);
  context.configure({
    device,
    format: presentationFormat,
    size: presentationSize,
  });

  const spriteShaderModule = device.createShaderModule({
    code: `
@stage(vertex)
fn vert_main(@location(0) a_particlePos: vec2<f32>,
        @location(1) a_particleVel : vec2<f32>,
        @location(2) a_pos: vec2<f32>
) -> @builtin(position) vec4<f32> {
    let angle = -atan2(a_particleVel.x, a_particleVel.y);
    let pos = vec2<f32>(
        (a_pos.x * cos(angle)) - (a_pos.y * sin(angle)),
        (a_pos.x * sin(angle)) + (a_pos.y * cos(angle))
    );
    return vec4<f32>(a_pos + a_particlePos, 0.0, 1.0);
}

@stage(fragment)
fn frag_main() -> @location(0) vec4<f32> {
    return vec4<f32>(1.0, 1.0, 1.0, 1.0);
}
      `,
  });

  const renderPipeline = device.createRenderPipeline({
    vertex: {
      module: spriteShaderModule,
      entryPoint: "vert_main",
      buffers: [
        {
          arrayStride: 4 * 4,
          stepMode: "instance",
          attributes: [
            {
              shaderLocation: 0,
              offset: 0,
              format: "float32x2",
            },
            {
              shaderLocation: 1,
              offset: 2 * 4,
              format: "float32x2",
            },
          ],
        },
        {
          arrayStride: 2 * 4,
          stepMode: "vertex",
          attributes: [
            {
              shaderLocation: 2,
              offset: 0,
              format: "float32x2",
            },
          ],
        },
      ],
    },
    fragment: {
      module: spriteShaderModule,
      entryPoint: "frag_main",
      targets: [{ format: presentationFormat }],
    },
    primitive: {
      topology: "triangle-list",
    },
  });

  const updateSpritesModule = device.createShaderModule({
    code: `
struct Particle {
    pos: vec2<f32>;
    vel: vec2<f32>;
};
struct SimParams {
    deltaT: f32;
    rule1Distance: f32;
    rule2Distance: f32;
    rule3Distance: f32;
    rule1Scale: f32;
    rule2Scale: f32;
    rule3Scale: f32;
};

struct Particles {
    particles: array<Particle>;
}

@group(0) @binding(0) var<uniform> params: SimParams;
@group(0) @binding(1) var<storage, read> particlesA: Particles;
@group(0) @binding(2) var<storage, read_write> particlesB: Particles;

@stage(compute) @workgroup_size(64) // TODO: あとで外しても動くか確かめる
fn main(@builtin(global_invocation_id) GlobalInvocationID : vec3<u32>) {
    var index : u32 = GlobalInvocationID.x;
    var vPos = particlesA.particles[index].pos;
    var vVel = particlesA.particles[index].vel;
    var cMass = vec2<f32>(0.0, 0.0);
    var cVel = vec2<f32>(0.0, 0.0);
    var colVel = vec2<f32>(0.0, 0.0);
    var cMassCount : u32 = 0u;
    var cVelCount : u32 = 0u;

    var pos : vec2<f32>;
    var vel : vec2<f32>;
    for (var i : u32 = 0u; i < arrayLength(&particlesA.particles); i = i + 1u) {
        if (i == index) {
            continue;
        }

        pos = particlesA.particles[i].pos.xy;
        vel = particlesA.particles[i].vel.xy;
        if (distance(pos, vPos) < params.rule1Distance) {
            cMass = cMass + pos;
            cMassCount = cMassCount + 1u;
        }
        if (distance(pos, vPos) < params.rule2Distance) {
            colVel = colVel - (pos - vPos);
        }
        if (distance(pos, vPos) < params.rule3Distance) {
            cVel = cVel + vel;
            cVelCount = cVelCount + 1u;
        }
    }
    if (cMassCount > 0u) {
        var temp = f32(cMassCount);
        cMass = (cMass / vec2<f32>(temp, temp)) - vPos;
    }
    if (cVelCount > 0u) {
        var temp = f32(cVelCount);
        cVel = cVel / vec2<f32>(temp, temp);
    }
    vVel = vVel + (cMass * params.rule1Scale) + (colVel * params.rule2Scale) + (cVel * params.rule3Scale);
    // clamp velocity for a more pleasing simulation
    vVel = normalize(vVel) * clamp(length(vVel), 0.0, 0.1);
    // kinematic update
    vPos = vPos + (vVel * params.deltaT);
    // wrap around boundary
    if (vPos.x < -1.0) {
        vPos.x = 1.0;
    }
    if (vPos.x > 1.0) {
        vPos.x = -1.0;
    }
    if(vPos.y < -1.0) {
        vPos.y = 1.0;
    }
    if(vPos.y > 1.0) {
        vPos.y = -1.0;
    }
    // Write back
    particlesB.particles[index].pos = vPos;
    particlesB.particles[index].vel = vVel;
}
      `,
  });

  const computePipeline = device.createComputePipeline({
    compute: {
      module: updateSpritesModule,
      entryPoint: "main",
    },
  });

  const renderPassDescriptor: GPURenderPassDescriptor = {
    colorAttachments: [
      {
        view: undefined, // Asigned later
        loadValue: [0, 0, 0, 1],
        storeOp: "store",
      },
    ],
  };

  const vertexBufferData = new Float32Array([
    -0.01, -0.02, 0.01, -0.02, 0.0, 0.02,
  ]);

  const spriteVertexBuffer = device.createBuffer({
    size: vertexBufferData.byteLength,
    usage: GPUBufferUsage.VERTEX,
    mappedAtCreation: true,
  });
  new Float32Array(spriteVertexBuffer.getMappedRange()).set(vertexBufferData);
  spriteVertexBuffer.unmap();

  const simParams = {
    deltaT: 0.04,
    rule1Distance: 0.1,
    rule2Distance: 0.025,
    rule3Distance: 0.025,
    rule1Scale: 0.02,
    rule2Scale: 0.05,
    rule3Scale: 0.005,
  };

  const simParamsBufferSize = 7 * Float32Array.BYTES_PER_ELEMENT;
  const simParamBuffer = device.createBuffer({
    size: simParamsBufferSize,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });

  function updateSimParams() {
    device.queue.writeBuffer(
      simParamBuffer,
      0,
      new Float32Array([
        simParams.deltaT,
        simParams.rule1Distance,
        simParams.rule2Distance,
        simParams.rule3Distance,
        simParams.rule1Scale,
        simParams.rule2Scale,
        simParams.rule3Scale,
      ])
    );
  }

  updateSimParams();

  const numParticles = 1500;
  const initialParticleData = new Float32Array(numParticles * 4);
  for (let i = 0; i < numParticles; ++i) {
    initialParticleData[4 * i + 0] = 2 * (Math.random() - 0.5);
    initialParticleData[4 * i + 1] = 2 * (Math.random() - 0.5);
    initialParticleData[4 * i + 2] = 2 * (Math.random() - 0.5) * 0.1;
    initialParticleData[4 * i + 3] = 2 * (Math.random() - 0.5) * 0.1;
  }

  const particleBuffers: GPUBuffer[] = new Array(2);
  const particleBindGroups: GPUBindGroup[] = new Array(2);
  for (let i = 0; i < 2; i++) {
    particleBuffers[i] = device.createBuffer({
      size: initialParticleData.byteLength,
      usage: GPUBufferUsage.VERTEX | GPUBufferUsage.STORAGE,
      mappedAtCreation: true,
    });
    new Float32Array(particleBuffers[i].getMappedRange()).set(
      initialParticleData
    );
    particleBuffers[i].unmap();
  }

  for (let i = 0; i < 2; i++) {
    particleBindGroups[i] = device.createBindGroup({
      layout: computePipeline.getBindGroupLayout(0),
      entries: [
        {
          binding: 0,
          resource: {
            buffer: simParamBuffer,
          },
        },
        {
          binding: 1,
          resource: {
            buffer: particleBuffers[i],
            offset: 0,
            size: initialParticleData.byteLength,
          },
        },
        {
          binding: 2,
          resource: {
            buffer: particleBuffers[(i + 1) % 2],
            offset: 0,
            size: initialParticleData.byteLength,
          },
        },
      ],
    });
  }

  let t = 0;
  function frame() {
    renderPassDescriptor.colorAttachments[0].view = context
      .getCurrentTexture()
      .createView();

    const commandEncoder = device.createCommandEncoder();
    {
      const passEncoder = commandEncoder.beginComputePass();
      passEncoder.setPipeline(computePipeline);
      passEncoder.setBindGroup(0, particleBindGroups[t % 2]);
      passEncoder.dispatch(Math.ceil(numParticles / 64));
      passEncoder.endPass();
    }
    {
      const passEncoder = commandEncoder.beginRenderPass(renderPassDescriptor);
      passEncoder.setPipeline(renderPipeline);
      passEncoder.setVertexBuffer(0, particleBuffers[(t + 1) % 2]);
      passEncoder.setVertexBuffer(1, spriteVertexBuffer);
      passEncoder.draw(3, numParticles, 0, 0);
      passEncoder.endPass();
    }
    device.queue.submit([commandEncoder.finish()]);
    ++t;
    requestAnimationFrame(frame);
  }

  requestAnimationFrame(frame);
}
