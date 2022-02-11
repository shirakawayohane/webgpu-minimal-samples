import { vec3 } from "gl-matrix";
import dragonRawData from "stanford-dragon/4";

// Compute surface normals
// mesh.normals = computeSurfaceNormals(mesh.positions, mesh.triangles)

// Compute some easy uvs for testing

function computeSurfaceNormals(positions: [number, number, number][], triangles: [number, number, number][]): [number, number, number][] {
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
    normals.forEach(n => {
        // Normalize accumulated normals.
        vec3.normalize(n, n);
    });

    return normals;
}

type ProjectedPlane = 'xy' | 'xz' | 'yz';

const projectedPlane2Ids: { [a in ProjectedPlane ] : [number, number] } = {
    xy: [0, 1],
    xz: [0, 2],
    yz: [1, 2]
}

function computeProjectedPlaneUVs(positions: [number, number, number][], projectedPlane: ProjectedPlane = "xy") {
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
    triangles: dragonRawData.cells as [number,number,number][],
    normals: [] as [number, number, number][],
    uvs: [] as [number, number][]
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
    [ 100, 20,  100],
    [-100, 20,  100],
    [ 100, 20, -100]
)

mesh.normals.push(
    [0, 1, 0],
    [0, 1, 0],
    [0, 1, 0],
    [0, 1, 0]
);

mesh.uvs.push(
    [0, 0],
    [1, 1,],
    [0, 1],
    [1, 0]
);

export async function init() {
    const adapter = await navigator.gpu.requestAdapter();
    const device = await adapter.requestDevice();
    const canvas = document.querySelector("canvas");
    const context = canvas.getContext("webgpu");
    const devicePixelRatio = window.devicePixelRatio || 1;
    const presentationSize = [
        canvas.clientWidth * devicePixelRatio,
        canvas.clientHeight * devicePixelRatio
    ];
    const aspect = presentationSize[0] / presentationSize[1];
    const presentationFormat = context.getPreferredFormat(adapter);
    context.configure({
        device,
        format: presentationFormat,
        size: presentationSize
    });

    // Create the model vertex buffer.
    const vertexBuffer = device.createBuffer({
        size: mesh.positions.length * 3 * 2 * Float32Array.BYTES_PER_ELEMENT,
        usage: GPUBufferUsage.VERTEX,
        mappedAtCreation: true
    });
    {
        const mapping = new Float32Array(vertexBuffer.getMappedRange());
        for(let i = 0; i < mesh.positions.length; ++i) {
            mapping.set(mesh.positions[i], 6 * i);
            mapping.set(mesh.positions[i], 6 * i + 3);
        }
        vertexBuffer.unmap();
    }
    // Create the depth texture for rendering/sampling the shadow map.
    const shadowDepthTexture = device.createTexture({
        size: [1024, 1024, 1],
        usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
        format: "depth32float"
    });
    const shadowDepthTextureView = shadowDepthTexture.createView();

    // Create some common descriptors used for both the shadow pipeline
    // and the color rendering pipeline.
    const vertexBuffers: Iterable<GPUVertexBufferLayout> = [
        {
            arrayStride: Float32Array.BYTES_PER_ELEMENT * 6,
            attributes: [
                {
                    shaderLocation: 0,
                    offset: 0,
                    format: "float32x3"
                },
                {
                    shaderLocation: 1,
                    offset: Float32Array.BYTES_PER_ELEMENT * 3,
                    format: "float32x3"
                }
            ]
        }
    ]

    const primitive: GPUPrimitiveState = {
        topology: "triangle-list",
        cullMode: "back"
    }

    const uniformBufferBindGroupLayout = device.createBindGroupLayout({
        entries: [
            {
                binding: 0,
                visibility: GPUShaderStage.VERTEX,
                buffer: {
                    type: "uniform"
                }
            }
        ]
    });

    
    
}