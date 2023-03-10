#version 450

layout (push_constant) uniform Uniforms {
    float invAspect;
    float renderScale;
    float renderOffsX, renderOffsZ;
} uniforms;

layout (location = 0) out vec3 out_vColor;

// per vertex attributes
layout (location = 0) in vec2 in_vPos;

// per instance attributes
layout (location = 1) in vec2 in_iPos;
layout (location = 2) in vec3 in_iColor;
layout (location = 3) in float in_iRadius;

layout (location = 6) out vec2 circlePos;

void main() {
    out_vColor = in_iColor;

    vec2 position = in_vPos * in_iRadius - vec2(uniforms.renderOffsX, uniforms.renderOffsZ);
    gl_Position = vec4(vec3((position + in_iPos) * vec2(uniforms.invAspect, 1.0) * uniforms.renderScale, 0.0), 1.0);
    circlePos = in_vPos * 2;
}
