#version 450

layout (location = 0) out vec4 fragColor;

layout (location = 0) in vec3 color;

layout (location = 6) in float centerDistance;

void main() {
    fragColor = vec4(color * centerDistance, 1.0);
}
