// todo: look for the next todo that is where the path is 
// you need to get https://github.com/TheThinMatrix/OpenGL-Animation/tree/master/Resources
// "You must specify your own path for where you would like your resources to be."
// Note: This code originated from [OpenGL skeletal animation in C++ using Assimp]
// [Coding Man (Itan)] on YouTube.
// It has been modified in some parts to use GLFW3 instead of SDL.
// His original code didn't seem to work, but the underlying premise remains the same.
//    ThinMatrix creates tutorials using Java, including an OpenGL Animation Tutorial and other fascinating content.
// Alongside other resources like Ogldev on YouTube, it is truly inspiring.
//There are more advanced tutorials available online, such as Learn OpenGL.
// https://learnopengl.com/Getting-started/OpenGL
// The youtube channel Thin uses is [Sebastian Lague] HE SHOWS HOW YOU CAN USE BLENDER TO BUILD SAID CHARACTER.

// ----------------------------------------------------------------------
// This version supports arbitrary bone influences per vertex using
// Shader Storage Buffer Objects (SSBOs). SSBOs allow each vertex to
// have as many bone influences as the model provides, removing the classic
// 4-bone-per-vertex limit, which is crucial for accurate character animation.
// ----------------------------------------------------------------------

#include <assimp/scene.h>          // Contains aiScene, aiMesh, aiNode, etc.
#include <assimp/postprocess.h>    // Contains post-processing flags like aiProcess_Triangulate
#include <assimp/Importer.hpp>     // For Assimp::Importer class

#include <glm/glm.hpp>                     // Basic GLM types (vec3, mat4)
#include <glm/gtc/matrix_transform.hpp>    // For glm::translate, glm::scale, glm::perspective, glm::lookAt
#include <glm/gtc/type_ptr.hpp>            // For glm::value_ptr (to pass matrices to OpenGL)
#include <glm/gtx/quaternion.hpp>          // For glm::quat operations like slerp

#include <unordered_map>    // For std::unordered_map
#include <vector>           // For std::vector
#include <string>           // For std::string
#include <iostream>         // For std::cout, std::cerr, std::endl
#include <algorithm>        // For std::max, std::min
#include <cmath>            // For std::fmod

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#include <glad/glad.h>
#include <GLFW/glfw3.h>

typedef unsigned int uint;
typedef unsigned char byte;

// ----------------------------------------------------------------------
// Converts an Assimp 4x4 matrix to a GLM 4x4 matrix.
// Assimp uses row-major by default, GLM uses column-major.
// This conversion handles the transpose so the data matches GLM's expectation.
// ----------------------------------------------------------------------
inline glm::mat4 assimpToGlmMatrix(const aiMatrix4x4& mat) {
    glm::mat4 m;
    for (int y = 0; y < 4; y++)
        for (int x = 0; x < 4; x++)
            m[x][y] = mat[y][x];
    return m;
}

// ----------------------------------------------------------------------
// Converts an Assimp 3D vector to a GLM 3D vector.
// ----------------------------------------------------------------------
inline glm::vec3 assimpToGlmVec3(const aiVector3D& v) {
    return glm::vec3(v.x, v.y, v.z);
}

// ----------------------------------------------------------------------
// Converts an Assimp quaternion to a GLM quaternion.
// Assimp and GLM quaternions have the same component order (w, x, y, z).
// ----------------------------------------------------------------------
inline glm::quat assimpToGlmQuat(const aiQuaternion& q) {
    return glm::quat(q.w, q.x, q.y, q.z);
}

// ----------------------------------------------------------------------
// Vertex Structure
// Instead of a fixed-size array for bone IDs/weights, each vertex stores:
// - influencesOffset: where in the global flat arrays its influences start
// - influencesCount: how many bone influences this vertex has
// The actual indices/weights are stored in global flat arrays used by the SSBOs.
// ----------------------------------------------------------------------
struct Vertex {
    glm::vec3 position;
    glm::vec3 normal;
    glm::vec2 uv;
    int influencesOffset;
    int influencesCount;
};

// ----------------------------------------------------------------------
// Bone structure representing a node in the skeleton hierarchy.
// ----------------------------------------------------------------------
struct Bone {
    int id = 0;
    std::string name;
    glm::mat4 offset = glm::mat4(1.0f);
    std::vector<Bone> children;
};

// ----------------------------------------------------------------------
// Stores animation keyframe data for a single bone.
// ----------------------------------------------------------------------
struct BoneTransformTrack {
    std::vector<float> positionTimestamps, rotationTimestamps, scaleTimestamps;
    std::vector<glm::vec3> positions, scales;
    std::vector<glm::quat> rotations;
};
// Contains overall animation information, including duration and bone tracks.
struct Animation {
    float duration = 0.0f, ticksPerSecond = 1.0f;
    std::unordered_map<std::string, BoneTransformTrack> boneTransforms;
};

// ----------------------------------------------------------------------
// Initializes GLFW, creates a window, and sets up OpenGL context.
// Returns a pointer to the GLFWwindow or nullptr on failure.
// ----------------------------------------------------------------------
GLFWwindow* glfwInitWindow(int& windowWidth, int& windowHeight) {
    if (!glfwInit()) {
        std::cerr << "Failed to initialize GLFW\n";
        return nullptr;
    }
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3); // SSBOs require OpenGL 4.3+
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
#ifdef __APPLE__
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
#endif
    GLFWwindow* window = glfwCreateWindow(windowWidth, windowHeight, "Skinning-Anim-SSBO", nullptr, nullptr);
    if (!window) {
        std::cerr << "Failed to create GLFW window\n";
        glfwTerminate();
        return nullptr;
    }
    glfwMakeContextCurrent(window);
    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
        std::cerr << "Failed to initialize GLAD\n";
        glfwTerminate();
        return nullptr;
    }
    glfwGetFramebufferSize(window, &windowWidth, &windowHeight);
    glViewport(0, 0, windowWidth, windowHeight);
    glEnable(GL_DEPTH_TEST);
    glfwSwapInterval(1);
    return window;
}

// ----------------------------------------------------------------------
// Compiles and links an OpenGL shader program from source strings.
// Returns the shader program ID or 0 on failure, printing errors to console.
// ----------------------------------------------------------------------
inline unsigned int createShader(const char* vertexStr, const char* fragmentStr) {
    int success;
    char info_log[512];
    GLuint program = glCreateProgram();
    GLuint vShader = glCreateShader(GL_VERTEX_SHADER);
    GLuint fShader = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(vShader, 1, &vertexStr, nullptr);
    glCompileShader(vShader);
    glGetShaderiv(vShader, GL_COMPILE_STATUS, &success);
    if (!success) {
        glGetShaderInfoLog(vShader, 512, nullptr, info_log);
        std::cerr << "Vertex shader compilation failed!\n" << info_log << std::endl;
        return 0;
    }
    glShaderSource(fShader, 1, &fragmentStr, nullptr);
    glCompileShader(fShader);
    glGetShaderiv(fShader, GL_COMPILE_STATUS, &success);
    if (!success) {
        glGetShaderInfoLog(fShader, 512, nullptr, info_log);
        std::cerr << "Fragment shader compilation failed!\n" << info_log << std::endl;
        return 0;
    }
    glAttachShader(program, vShader);
    glAttachShader(program, fShader);
    glLinkProgram(program);
    glGetProgramiv(program, GL_LINK_STATUS, &success);
    if (!success) {
        glGetProgramInfoLog(program, 512, nullptr, info_log);
        std::cerr << "Program linking failed!\n" << info_log << std::endl;
        return 0;
    }
    glDeleteShader(vShader);
    glDeleteShader(fShader);
    return program;
}

// ----------------------------------------------------------------------
// Loads an image from filepath and creates an OpenGL 2D texture.
// Returns the texture ID or 0 on failure.
// ----------------------------------------------------------------------
unsigned int createTexture(const std::string& filepath) {
    unsigned int tex;
    int width, height, nrChannels;
    stbi_set_flip_vertically_on_load(true); // Flip texture vertically for OpenGL
    unsigned char* data = stbi_load(filepath.c_str(), &width, &height, &nrChannels, 4);
    if (!data) {
        std::cerr << "Failed to load texture: " << filepath << std::endl;
        return 0;
    }
    glGenTextures(1, &tex);
    glBindTexture(GL_TEXTURE_2D, tex);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR); // Use mipmaps
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, data);
    glGenerateMipmap(GL_TEXTURE_2D); // Generate mipmaps after uploading data
    stbi_image_free(data); // Free image memory after creating OpenGL texture
    glBindTexture(GL_TEXTURE_2D, 0); // Unbind texture
    return tex;
}

// ----------------------------------------------------------------------
// Recursively reads the bone hierarchy from Assimp's scene graph.
// It populates the 'boneOutput' structure with bone ID, name, offset matrix, and children.
// 'boneInfoTable' maps bone names to their ID and offset matrix.
// ----------------------------------------------------------------------
bool readSkeleton(Bone& boneOutput, aiNode* node,
    const std::unordered_map<std::string, std::pair<int, glm::mat4>>& boneInfoTable) {
    auto it = boneInfoTable.find(node->mName.C_Str());
    if (it != boneInfoTable.end()) {
        boneOutput.name = node->mName.C_Str();
        boneOutput.id = it->second.first;
        boneOutput.offset = it->second.second;
        for (unsigned int i = 0; i < node->mNumChildren; i++) {
            Bone child;
            readSkeleton(child, node->mChildren[i], boneInfoTable);
            boneOutput.children.push_back(child);
        }
        return true;
    } else {
        for (unsigned int i = 0; i < node->mNumChildren; i++)
            if (readSkeleton(boneOutput, node->mChildren[i], boneInfoTable))
                return true;
    }
    return false;
}

// ----------------------------------------------------------------------
// Loads mesh data (vertices, indices, bone weights) and builds the skeleton.
// Instead of storing only 4 bone influences per vertex, we gather all influences
// and store them in flat arrays for use with SSBOs.
// Reports the maximum number of bone influences per vertex for diagnostics.
// ----------------------------------------------------------------------
void loadModel(
    const aiScene* scene, aiMesh* mesh,
    std::vector<Vertex>& verticesOut, std::vector<uint>& indicesOut,
    Bone& skeletonOut, uint& boneCount,
    std::vector<int>& boneIndicesSSBO, std::vector<float>& boneWeightsSSBO,
    int& maxBonesPerVertex // Output: for diagnostics
) {
    verticesOut.clear(); indicesOut.clear(); boneIndicesSSBO.clear(); boneWeightsSSBO.clear();
    verticesOut.reserve(mesh->mNumVertices);
    std::unordered_map<std::string, std::pair<int, glm::mat4>> boneInfoTable;
    std::vector<std::vector<std::pair<int, float>>> perVertexInfluences(mesh->mNumVertices);
    boneCount = mesh->mNumBones;

    // Loop through each bone in the mesh
    for (unsigned int i = 0; i < boneCount; i++) {
        const aiBone* bone = mesh->mBones[i];
        boneInfoTable[bone->mName.C_Str()] = { int(i), assimpToGlmMatrix(bone->mOffsetMatrix) };
        for (unsigned int j = 0; j < bone->mNumWeights; j++) {
            unsigned int vId = bone->mWeights[j].mVertexId;
            float w = bone->mWeights[j].mWeight;
            perVertexInfluences[vId].emplace_back(i, w);
        }
    }
    maxBonesPerVertex = 0;
    // For each vertex, pack all bone influences into the global flat arrays
    for (unsigned int i = 0; i < mesh->mNumVertices; i++) {
        Vertex v{};
        v.position = assimpToGlmVec3(mesh->mVertices[i]);
        v.normal = assimpToGlmVec3(mesh->mNormals[i]);
        v.uv = mesh->mTextureCoords[0] ? glm::vec2(mesh->mTextureCoords[0][i].x, mesh->mTextureCoords[0][i].y)
                                       : glm::vec2(0.0f);
        float total = 0.0f;
        for (auto& p : perVertexInfluences[i]) total += p.second;
        // Normalize weights for each vertex to ensure smooth skinning
        if (total > 0.0f) for (auto& p : perVertexInfluences[i]) p.second /= total;

        v.influencesOffset = static_cast<int>(boneIndicesSSBO.size());
        v.influencesCount = static_cast<int>(perVertexInfluences[i].size());
        maxBonesPerVertex = std::max(maxBonesPerVertex, v.influencesCount);
        for (auto& p : perVertexInfluences[i]) {
            boneIndicesSSBO.push_back(p.first);
            boneWeightsSSBO.push_back(p.second);
        }
        verticesOut.push_back(v);
    }
    // Load indices (face data)
    for (unsigned int i = 0; i < mesh->mNumFaces; i++) {
        const aiFace& face = mesh->mFaces[i];
        for (unsigned int j = 0; j < face.mNumIndices; j++)
            indicesOut.push_back(face.mIndices[j]);
    }
    // Build the bone hierarchy
    readSkeleton(skeletonOut, scene->mRootNode, boneInfoTable);

    // --- Diagnostic Output: Max number of bones per vertex ---
    std::cout << "Max bones per vertex in this mesh: " << maxBonesPerVertex << std::endl;
    if (maxBonesPerVertex > 4) {
        std::cout << "NOTE: This mesh has vertices influenced by more than 4 bones! "
                     "Using SSBOs allows correct skinning with all influences preserved.\n";
    }
}

// ----------------------------------------------------------------------
// Loads animation data from the Assimp scene into the Animation structure.
// Reads positions, rotations, and scales for each animated bone.
// ----------------------------------------------------------------------
void loadAnimation(const aiScene* scene, Animation& animation) {
    if (scene->mNumAnimations == 0) return;
    const aiAnimation* anim = scene->mAnimations[0];
    animation.ticksPerSecond = anim->mTicksPerSecond != 0.0 ? float(anim->mTicksPerSecond) : 1.0f;
    animation.duration = float(anim->mDuration);
    animation.boneTransforms.clear();
    for (unsigned int i = 0; i < anim->mNumChannels; i++) {
        const aiNodeAnim* channel = anim->mChannels[i];
        BoneTransformTrack track;
        for (unsigned int j = 0; j < channel->mNumPositionKeys; j++) {
            track.positionTimestamps.push_back(float(channel->mPositionKeys[j].mTime));
            track.positions.push_back(assimpToGlmVec3(channel->mPositionKeys[j].mValue));
        }
        for (unsigned int j = 0; j < channel->mNumRotationKeys; j++) {
            track.rotationTimestamps.push_back(float(channel->mRotationKeys[j].mTime));
            track.rotations.push_back(assimpToGlmQuat(channel->mRotationKeys[j].mValue));
        }
        for (unsigned int j = 0; j < channel->mNumScalingKeys; j++) {
            track.scaleTimestamps.push_back(float(channel->mScalingKeys[j].mTime));
            track.scales.push_back(assimpToGlmVec3(channel->mScalingKeys[j].mValue));
        }
        animation.boneTransforms[channel->mNodeName.C_Str()] = track;
    }
}

// ----------------------------------------------------------------------
// Helper function to find the animation segment and interpolation fraction.
// Given a current time 'currentTimeTicks' (in animation ticks, normalized to loop)
// and a vector of 'times' (keyframe timestamps in ticks),
// it returns the index of the START keyframe and the interpolation fraction.
// ----------------------------------------------------------------------
std::pair<unsigned int, float> getTimeFraction(const std::vector<float>& times, float currentTimeTicks) {
    if (times.empty()) return { 0, 0.0f };
    unsigned int segment = 0;
    while (segment < times.size() - 1 && currentTimeTicks >= times[segment + 1]) segment++;
    if (segment >= times.size() - 1)
        return { static_cast<unsigned int>(times.size() - 1), 1.0f };
    float startTime = times[segment], endTime = times[segment + 1];
    float frac = (endTime - startTime) > 0.0f ? (currentTimeTicks - startTime) / (endTime - startTime) : 0.0f;
    return { segment, frac };
}

// ----------------------------------------------------------------------
// Recursively calculates the bone matrices for the current animation pose.
// 'output' vector will contain the final matrices to be uploaded to the shader.
// 'parentTransform' is the accumulated transform of the parent bone in world space.
// 'globalInverseTransform' is the inverse of the root node's initial transform,
// used to bring the final bone transform into the model's original coordinate system.
// ----------------------------------------------------------------------
void getPose(const Animation& animation, const Bone& skeleton, float currentTimeTicks,
    std::vector<glm::mat4>& output, const glm::mat4& parentTransform,
    const glm::mat4& globalInverseTransform) {

    auto it = animation.boneTransforms.find(skeleton.name);
    glm::mat4 localTransform = glm::mat4(1.0f);
    if (it != animation.boneTransforms.end()) {
        const BoneTransformTrack& btt = it->second;
        glm::vec3 finalPosition = btt.positions.empty() ? glm::vec3(0.0f) : btt.positions[0];
        glm::quat finalRotation = btt.rotations.empty() ? glm::quat(1.0f, 0, 0, 0) : btt.rotations[0];
        glm::vec3 finalScale = btt.scales.empty() ? glm::vec3(1.0f) : btt.scales[0];

        if (btt.positions.size() > 1) {
            auto [idx, f] = getTimeFraction(btt.positionTimestamps, currentTimeTicks);
            unsigned int next = std::min(idx + 1, unsigned(btt.positions.size() - 1));
            finalPosition = glm::mix(btt.positions[idx], btt.positions[next], f);
        }
        if (btt.rotations.size() > 1) {
            auto [idx, f] = getTimeFraction(btt.rotationTimestamps, currentTimeTicks);
            unsigned int next = std::min(idx + 1, unsigned(btt.rotations.size() - 1));
            finalRotation = glm::slerp(btt.rotations[idx], btt.rotations[next], f);
        }
        if (btt.scales.size() > 1) {
            auto [idx, f] = getTimeFraction(btt.scaleTimestamps, currentTimeTicks);
            unsigned int next = std::min(idx + 1, unsigned(btt.scales.size() - 1));
            finalScale = glm::mix(btt.scales[idx], btt.scales[next], f);
        }
        glm::mat4 positionMat = glm::translate(glm::mat4(1.0f), finalPosition);
        glm::mat4 rotationMat = glm::toMat4(finalRotation);
        glm::mat4 scaleMat = glm::scale(glm::mat4(1.0f), finalScale);
        localTransform = positionMat * rotationMat * scaleMat;
    }
    glm::mat4 globalTransform = parentTransform * localTransform;
    if (static_cast<size_t>(skeleton.id) < output.size())
        output[skeleton.id] = globalInverseTransform * globalTransform * skeleton.offset;
    for (const Bone& child : skeleton.children)
        getPose(animation, child, currentTimeTicks, output, globalTransform, globalInverseTransform);
}

// ----------------------------------------------------------------------
// Creates and configures a Vertex Array Object (VAO) for the model.
// Binds vertex attributes (position, normal, UV, influencesOffset, influencesCount) to buffer data.
// ----------------------------------------------------------------------
unsigned int createVertexArray(const std::vector<Vertex>& vertices, const std::vector<uint>& indices) {
    GLuint vao, vbo, ebo;
    glGenVertexArrays(1, &vao);
    glGenBuffers(1, &vbo);
    glGenBuffers(1, &ebo);
    glBindVertexArray(vao);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, vertices.size()*sizeof(Vertex), vertices.data(), GL_STATIC_DRAW);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.size()*sizeof(uint), indices.data(), GL_STATIC_DRAW);

    glEnableVertexAttribArray(0); // position
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)offsetof(Vertex, position));
    glEnableVertexAttribArray(1); // normal
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)offsetof(Vertex, normal));
    glEnableVertexAttribArray(2); // uv
    glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)offsetof(Vertex, uv));
    glEnableVertexAttribArray(3); // influencesOffset
    glVertexAttribIPointer(3, 1, GL_INT, sizeof(Vertex), (void*)offsetof(Vertex, influencesOffset));
    glEnableVertexAttribArray(4); // influencesCount
    glVertexAttribIPointer(4, 1, GL_INT, sizeof(Vertex), (void*)offsetof(Vertex, influencesCount));
    glBindVertexArray(0);
    return vao;
}

// ----------------------------------------------------------------------
// Vertex Shader (GLSL) for SSBO-based skinning.
// For each vertex, fetches bone indices and weights from the global SSBO arrays,
// as specified by influencesOffset/influencesCount. This allows any number of
// bone influences per vertex, removing the classic 4-bone-per-vertex limit.
// ----------------------------------------------------------------------
const char* vertexShaderSource = R"(#version 430 core
layout(location = 0) in vec3 position;
layout(location = 1) in vec3 normal;
layout(location = 2) in vec2 uv;
layout(location = 3) in int influencesOffset;
layout(location = 4) in int influencesCount;

layout(std430, binding = 0) buffer BoneIndicesBuffer { int indices[]; };
layout(std430, binding = 1) buffer BoneWeightsBuffer { float weights[]; };

uniform mat4 bone_transforms[100]; // adjust max as needed
uniform mat4 view_projection_matrix;
uniform mat4 model_matrix;

out vec2 tex_cord;
out vec3 v_normal;
out vec3 v_pos;

void main() {
    // Skinning: sum over all bone influences for this vertex using data from SSBOs
    vec4 animatedPos = vec4(0.0);
    for (int i = 0; i < influencesCount; ++i) {
        int idx = indices[influencesOffset + i];
        float w = weights[influencesOffset + i];
        animatedPos += w * (bone_transforms[idx] * vec4(position, 1.0));
    }
    gl_Position = view_projection_matrix * model_matrix * animatedPos;
    v_pos = vec3(model_matrix * animatedPos);
    tex_cord = uv;
    v_normal = normal; // (skinning for normals can be added here)
}
)";

// ----------------------------------------------------------------------
// Fragment shader: simply samples the diffuse texture for the current UV.
// ----------------------------------------------------------------------
const char* fragmentShaderSource = R"(#version 430 core
in vec2 tex_cord;
out vec4 color;
uniform sampler2D diff_texture;
void main() {
    vec3 dCol = texture(diff_texture, tex_cord).rgb;
    color = vec4(dCol, 1);
}
)";

// ----------------------------------------------------------------------
// Main Program
// ----------------------------------------------------------------------
int main() {
    int windowWidth = 800, windowHeight = 600;
    GLFWwindow* window = glfwInitWindow(windowWidth, windowHeight);
    if (!window) return -1;

    // todo: "You must specify your own path for where you would like your resources to be."
    // CHANGE THESE PATHS TO YOUR OWN MODEL AND TEXTURE!
    const std::string modelPath = "F:/res/models/man/model.dae";
    const std::string texturePath = "F:/res/models/man/diffuse.png";

    Assimp::Importer importer;
    const aiScene* scene = importer.ReadFile(modelPath.c_str(), aiProcess_Triangulate | aiProcess_FlipUVs | aiProcess_GenSmoothNormals);
    if (!scene || scene->mFlags & AI_SCENE_FLAGS_INCOMPLETE || !scene->mRootNode) {
        std::cerr << "ERROR::ASSIMP::" << importer.GetErrorString() << std::endl;
        glfwDestroyWindow(window); glfwTerminate(); return -1;
    }
    aiMesh* mesh = scene->mMeshes[0];
    std::vector<Vertex> vertices;
    std::vector<uint> indices;
    std::vector<int> boneIndicesSSBO;
    std::vector<float> boneWeightsSSBO;
    Bone skeleton;
    uint boneCount = 0;
    Animation animation;
    glm::mat4 globalInverseTransform = glm::inverse(assimpToGlmMatrix(scene->mRootNode->mTransformation));
    int maxBonesPerVertex = 0;
    loadModel(scene, mesh, vertices, indices, skeleton, boneCount, boneIndicesSSBO, boneWeightsSSBO, maxBonesPerVertex);
    loadAnimation(scene, animation);

    // --- OpenGL Buffer and Object Creation ---
    unsigned int vao = createVertexArray(vertices, indices);
    unsigned int diffuseTexture = createTexture(texturePath);
    unsigned int shader = createShader(vertexShaderSource, fragmentShaderSource);
    if (!shader) return -1;

    // ----------------------------------------------------------------------
    // SSBO setup
    // SSBOs ("Shader Storage Buffer Objects") are used here to remove the 4-bone-per-vertex limitation.
    // All bone indices and weights for all vertices are stored in flat arrays.
    // Each vertex only stores its offset and count into these arrays.
    // This allows models with any number of bone influences per vertex.
    // ----------------------------------------------------------------------
    GLuint ssboIndices, ssboWeights;
    glGenBuffers(1, &ssboIndices); glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssboIndices);
    glBufferData(GL_SHADER_STORAGE_BUFFER, boneIndicesSSBO.size()*sizeof(int), boneIndicesSSBO.data(), GL_STATIC_DRAW);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, ssboIndices);

    glGenBuffers(1, &ssboWeights); glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssboWeights);
    glBufferData(GL_SHADER_STORAGE_BUFFER, boneWeightsSSBO.size()*sizeof(float), boneWeightsSSBO.data(), GL_STATIC_DRAW);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, ssboWeights);

    GLint viewProjLoc = glGetUniformLocation(shader, "view_projection_matrix");
    GLint modelLoc = glGetUniformLocation(shader, "model_matrix");
    GLint boneLoc = glGetUniformLocation(shader, "bone_transforms");
    GLint texLoc = glGetUniformLocation(shader, "diff_texture");

    // Setup camera and model transformation matrices
    glm::mat4 projectionMatrix = glm::perspective(glm::radians(75.0f), float(windowWidth) / windowHeight, 0.01f, 1000.0f);
    glm::mat4 viewMatrix = glm::lookAt(glm::vec3(0, 0, -15), glm::vec3(0, 0, 0), glm::vec3(0, 1, 0));
    glm::mat4 viewProjectionMatrix = projectionMatrix * viewMatrix;
    glm::mat4 modelMatrix(1.0f);
    // Apply transformations in S-R-T order (Scale, Rotate, Translate)
    modelMatrix = glm::scale(modelMatrix, glm::vec3(1.0f));
    // Rotates 180 degrees around the X-axis to flip the model from upside down to upright
    modelMatrix = glm::rotate(modelMatrix, glm::radians(180.0f), glm::vec3(1.0f, 0.0f, 0.0f));
    // Rotates 180 degrees around the Y-axis to make the model face the camera
    modelMatrix = glm::rotate(modelMatrix, glm::radians(180.0f), glm::vec3(0.0f, 1.0f, 0.0f));
    modelMatrix = glm::translate(modelMatrix, glm::vec3(0.0f, -0.01f, 0.0f)); // Apply translation for ground alignment

    std::vector<glm::mat4> currentPose(boneCount, glm::mat4(1.0f));

    // ----------------------------------------------------------------------
    // Main Render Loop (with MSVC warning C4267 fix: correct type for GLsizei)
    // ----------------------------------------------------------------------
    while (!glfwWindowShouldClose(window)) {
        glfwPollEvents();
        float elapsedTime = float(glfwGetTime());
        // Calculate the total animation duration in SECONDS
        float animationDurationSeconds = animation.duration / animation.ticksPerSecond;
        // Calculate the current animation time in SECONDS, normalized to loop
        float loopedTimeSeconds = std::fmod(elapsedTime, animationDurationSeconds);
        // Convert the looped time (in seconds) to animation TICKS
        float loopedTimeTicks = loopedTimeSeconds * animation.ticksPerSecond;
        // Calculate bone matrices for the current animation pose
        getPose(animation, skeleton, loopedTimeTicks, currentPose, glm::mat4(1.0f), globalInverseTransform);

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        glUseProgram(shader);
        glUniformMatrix4fv(viewProjLoc, 1, GL_FALSE, glm::value_ptr(viewProjectionMatrix));
        glUniformMatrix4fv(modelLoc, 1, GL_FALSE, glm::value_ptr(modelMatrix));
        // FIX: Explicitly cast size_t to GLsizei to avoid MSVC warning C4267
        glUniformMatrix4fv(boneLoc, static_cast<GLsizei>(currentPose.size()), GL_FALSE, glm::value_ptr(currentPose[0]));
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, diffuseTexture);
        glUniform1i(texLoc, 0);
        glBindVertexArray(vao);
        // FIX: Explicitly cast size_t to GLsizei to avoid MSVC warning C4267
        glDrawElements(GL_TRIANGLES, static_cast<GLsizei>(indices.size()), GL_UNSIGNED_INT, nullptr);
        glfwSwapBuffers(window);
    }

    // ----------------------------------------------------------------------
    // Cleanup OpenGL objects and terminate GLFW
    // ----------------------------------------------------------------------
    glDeleteProgram(shader);
    glDeleteVertexArrays(1, &vao);
    glDeleteTextures(1, &diffuseTexture);
    glDeleteBuffers(1, &ssboIndices);
    glDeleteBuffers(1, &ssboWeights);
    glfwDestroyWindow(window);
    glfwTerminate();
    return 0;
}
