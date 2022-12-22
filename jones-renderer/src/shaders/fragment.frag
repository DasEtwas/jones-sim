#version 450

layout (location = 0) out vec4 fragColor;

layout (location = 0) in vec3 color;

layout (location = 6) in vec2 circlePos;

void main() {
    fragColor = vec4(color * dot(vec3(circlePos.x, sqrt(1.0 - dot(circlePos, circlePos)), circlePos.y), normalize(vec3(0.3, 0.6, 0.2))), 1.0);
}
