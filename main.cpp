// todo: look for the next todo that is where the path is 
// you need to get https://github.com/TheThinMatrix/OpenGL-Animation/tree/master/Resources
// "You must specify your own path for where you would like your resources to be."
// Note: This code originated from [OpenGL skeletal animation in C++ using Assimp]
// [Coding Man (Itan)] on YouTube.
// It has been modified in some parts to use GLFW3 instead of SDL.
// His original code didn't seem to work, but the underlying premise remains the same.



// No #pragma comment(lib, ...) directives here; these are for build system.

// Include necessary headers
#include <assimp/scene.h>          // Contains aiScene, aiMesh, aiNode, etc.
#include <assimp/postprocess.h>    // Contains post-processing flags like aiProcess_Triangulate
#include <assimp/Importer.hpp>     // For Assimp::Importer class

#include <glm/glm.hpp>                     // Basic GLM types (vec3, mat4)
#include <glm/gtc/matrix_transform.hpp>    // For glm::translate, glm::scale, glm::perspective, glm::lookAt
#include <glm/gtc/type_ptr.hpp>            // For glm::value_ptr (to pass matrices to OpenGL)
#include <glm/gtx/quaternion.hpp>          // For glm::quat operations like slerp
#include <glm/gtc/quaternion.hpp>          // For glm::quat itself (can sometimes be covered by gtx/quaternion)

#include <unordered_map>    // For std::unordered_map
#include <vector>           // For std::vector
#include <string>           // For std::string
#include <iostream>         // For std::cout, std::cerr, std::endl
#include <algorithm>        // For std::max, std::min
#include <cmath>            // For std::fmod

// STB_IMAGE_IMPLEMENTATION should ideally be in one .cpp file only,
// typically a dedicated image loader file. For a single-file example, it's placed here.
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

// GLAD and GLFW headers
#include <glad/glad.h>
#include <GLFW/glfw3.h>

// Type aliases for clarity and potential cross-platform consistency
typedef unsigned int uint;
typedef unsigned char byte;

// --- Assimp to GLM Conversion Helpers ---

// Converts an Assimp 4x4 matrix to a GLM 4x4 matrix.
// Assimp uses row-major by default, GLM uses column-major.
// This conversion handles the transpose.
inline glm::mat4 assimpToGlmMatrix(const aiMatrix4x4& mat) {
    glm::mat4 m;
    // Assimp stores matrices in row-major order, GLM uses column-major.
    // Direct assignment with [x][y] as mat[y][x] effectively transposes it.
    for (int y = 0; y < 4; y++) {
        for (int x = 0; x < 4; x++) {
            m[x][y] = mat[y][x];
        }
    }
    return m;
}

// Converts an Assimp 3D vector to a GLM 3D vector.
inline glm::vec3 assimpToGlmVec3(const aiVector3D& vec) {
    return glm::vec3(vec.x, vec.y, vec.z);
}

// Converts an Assimp quaternion to a GLM quaternion.
// Assimp and GLM quaternions have the same component order (w, x, y, z).
inline glm::quat assimpToGlmQuat(const aiQuaternion& quat) {
    return glm::quat(quat.w, quat.x, quat.y, quat.z);
}

// --- Shader Creation Function ---

// Creates and compiles an OpenGL shader program from vertex and fragment shader strings.
// Returns the shader program ID or 0 on failure, printing errors to console.
inline unsigned int createShader(const char* vertexStr, const char* fragmentStr) {
    int success;
    char info_log[512]; // Buffer for error messages

    // Create shader objects
    uint program = glCreateProgram();
    uint vShader = glCreateShader(GL_VERTEX_SHADER);
    uint fShader = glCreateShader(GL_FRAGMENT_SHADER);

    // Compile vertex shader
    glShaderSource(vShader, 1, &vertexStr, nullptr);
    glCompileShader(vShader);
    glGetShaderiv(vShader, GL_COMPILE_STATUS, &success);
    if (!success) {
        glGetShaderInfoLog(vShader, 512, nullptr, info_log);
        std::cerr << "Vertex shader compilation failed!\n" << info_log << std::endl;
        glDeleteShader(vShader);
        glDeleteShader(fShader);
        glDeleteProgram(program);
        return 0;
    }

    // Compile fragment shader
    glShaderSource(fShader, 1, &fragmentStr, nullptr);
    glCompileShader(fShader);
    glGetShaderiv(fShader, GL_COMPILE_STATUS, &success);
    if (!success) {
        glGetShaderInfoLog(fShader, 512, nullptr, info_log);
        std::cerr << "Fragment shader compilation failed!\n" << info_log << std::endl;
        glDeleteShader(vShader);
        glDeleteShader(fShader);
        glDeleteProgram(program);
        return 0;
    }

    // Link shaders into a program
    glAttachShader(program, vShader);
    glAttachShader(program, fShader);
    glLinkProgram(program);
    glGetProgramiv(program, GL_LINK_STATUS, &success);
    if (!success) {
        glGetProgramInfoLog(program, 512, nullptr, info_log);
        std::cerr << "Program linking failed!\n" << info_log << std::endl;
        glDeleteShader(vShader);
        glDeleteShader(fShader);
        glDeleteProgram(program);
        return 0;
    }

    // Detach and delete shaders as they are now linked into the program
    glDetachShader(program, vShader);
    glDeleteShader(vShader);
    glDetachShader(program, fShader);
    glDeleteShader(fShader);

    return program;
}

// --- GLFW Window Initialization ---

#define DO_GLFW3 // Using GLFW for window management

#ifdef DO_GLFW3
// Initializes GLFW, creates a window, and sets up OpenGL context.
// Returns a pointer to the GLFWwindow or nullptr on failure.
GLFWwindow* glfwInitWindow(int& windowWidth, int& windowHeight) {
    // Initialize GLFW
    if (!glfwInit()) {
        std::cerr << "Failed to initialize GLFW" << std::endl;
        return nullptr;
    }

    // Set OpenGL version hints for core profile
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 4);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

#ifdef __APPLE__
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE); // Required for macOS
#endif

    // Create window
    GLFWwindow* window = glfwCreateWindow(640, 480, "Skin Animation", nullptr, nullptr);
    if (!window) {
        std::cerr << "Failed to create GLFW window" << std::endl;
        glfwTerminate(); // Clean up GLFW
        return nullptr;
    }

    // Make the OpenGL context current on the created window
    glfwMakeContextCurrent(window);

    // Load OpenGL functions using GLAD
    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
        std::cerr << "Failed to initialize GLAD" << std::endl;
        glfwTerminate(); // Clean up GLFW
        return nullptr;
    }

    // Get window framebuffer size for viewport
    glfwGetFramebufferSize(window, &windowWidth, &windowHeight);
    glViewport(0, 0, windowWidth, windowHeight);

    // Set clear color and enable depth testing
    glClearColor(1.0f, 0.0f, 0.4f, 1.0f);
    glEnable(GL_DEPTH_TEST);

    // Enable vsync (1 for on, 0 for off)
    glfwSwapInterval(1);

    return window;
}
#endif // DO_GLFW3

// --- Shader Source Code ---
// This Shader was modified base from ...
//https://lisyarus.github.io/blog/posts/gltf-animation.html

// --- Shader Source Code ---
const char* vertexShaderSource = R"(
#version 440 core
layout (location = 0) in vec3 position;
layout (location = 1) in vec3 normal;
layout (location = 2) in vec2 uv;
layout (location = 3) in vec4 boneIds;      // IDs of bones influencing this vertex
layout (location = 4) in vec4 boneWeights;  // Weights corresponding to boneIds

out vec2 tex_cord;
out vec3 v_normal;
out vec3 v_pos;

uniform mat4 bone_transforms[50]; // Array of bone transformation matrices
uniform mat4 view_projection_matrix;
uniform mat4 model_matrix;

vec4 applyBoneTransform(vec4 p) {
    vec4 result = vec4(0.0);
    for (int i = 0; i < 4; ++i) {
        result += boneWeights[i] * (bone_transforms[int(boneIds[i])] * p);
    }
    return result;
}

void main() {
    vec4 animatedPos = applyBoneTransform(vec4(position, 1.0));
    vec3 animatedNormal = normalize(mat3(transpose(inverse(mat3(model_matrix * bone_transforms[int(boneIds.x)])))) * normal);

    // Final vertex position in clip space
    gl_Position = view_projection_matrix * model_matrix * animatedPos;

    // Pass transformed position to fragment shader
    v_pos = vec3(model_matrix * animatedPos);
    tex_cord = uv;
    v_normal = animatedNormal;
}
)";

/*
// This shader works also
// Vertex shader source string (raw string literal for multiline string)
const char* vertexShaderSource = R"(
#version 440 core
layout (location = 0) in vec3 position;
layout (location = 1) in vec3 normal;
layout (location = 2) in vec2 uv;
layout (location = 3) in vec4 boneIds;      // IDs of bones influencing this vertex
layout (location = 4) in vec4 boneWeights;  // Weights corresponding to boneIds

out vec2 tex_cord;
out vec3 v_normal;
out vec3 v_pos;

uniform mat4 bone_transforms[50]; // Array of bone transformation matrices
uniform mat4 view_projection_matrix;
uniform mat4 model_matrix;

void main()
{
    // Initialize boneTransform accumulator
    mat4 boneTransform = mat4(0.0);

    // Accumulate bone transformations weighted by bone weights.
    // boneIds are float but treated as integers to index the bone_transforms array.
    boneTransform += bone_transforms[int(boneIds.x)] * boneWeights.x;
    boneTransform += bone_transforms[int(boneIds.y)] * boneWeights.y;
    boneTransform += bone_transforms[int(boneIds.z)] * boneWeights.z;
    boneTransform += bone_transforms[int(boneIds.w)] * boneWeights.w;

    // Apply the accumulated bone transform to the vertex position
    vec4 animatedPos = boneTransform * vec4(position, 1.0);

    // Final vertex position in clip space
    gl_Position = view_projection_matrix * model_matrix * animatedPos;

    // Pass transformed position (in world space) to fragment shader
    v_pos = vec3(model_matrix * animatedPos);

    // Pass texture coordinates to fragment shader
    tex_cord = uv;

    // Calculate normal in world space for lighting
    // Transpose(inverse(model_matrix * boneTransform)) for correct normal transformation
    v_normal = mat3(transpose(inverse(model_matrix * boneTransform))) * normal;
    v_normal = normalize(v_normal); // Ensure normal is normalized
}
)";
*/

// Fragment shader source string
//const char* fragmentShaderSource = R"(
//#version 440 core
//
//in vec2 tex_cord;
//in vec3 v_normal;
//in vec3 v_pos;
//out vec4 color;
//
//uniform sampler2D diff_texture; // Diffuse texture sampler
//
//vec3 lightPos = vec3(0.2, 1.0, -3.0); // Simple directional light position
//
//void main()
//{
//    // Calculate light direction from fragment position to light source
//    vec3 lightDir = normalize(lightPos - v_pos);
//
//    // Calculate diffuse component using dot product of normal and light direction
//    // Clamped to a minimum of 0.2 to simulate ambient light
//    float diff = max(dot(v_normal, lightDir), 0.2);
//
//    // Sample texture and multiply by diffuse light component
//    vec3 dCol = diff * texture(diff_texture, tex_cord).rgb;
//
//    // Output final fragment color
//    color = vec4(dCol, 1);
//}
//)";
 
 // Fragment shader (temporary for testing)
const char* fragmentShaderSource = R"(
#version 440 core

in vec2 tex_cord;
// in vec3 v_normal; // Can comment out if only testing texture
// in vec3 v_pos;    // Can comment out if only testing texture
out vec4 color;

uniform sampler2D diff_texture;

// vec3 lightPos = vec3(0.2, 1.0, -3.0); // No longer needed

void main()
{
    // vec3 lightDir = normalize(lightPos - v_pos); // No longer needed
    // float diff = max(dot(v_normal, lightDir), 0.2); // No longer needed

    // Sample texture directly
    vec3 dCol = texture(diff_texture, tex_cord).rgb;

    color = vec4(dCol, 1);
}
)";



// --- Data Structures for Animated Model ---

// Represents a single vertex with position, normal, UV, bone IDs, and bone weights.
struct Vertex {
    glm::vec3 position;
    glm::vec3 normal;
    glm::vec2 uv;
    glm::vec4 boneIds = glm::vec4(0);       // Stores up to 4 bone IDs (as floats to match GLSL vec4)
    glm::vec4 boneWeights = glm::vec4(0.0f); // Stores up to 4 bone weights
};

// Represents a bone in the skeleton hierarchy.
struct Bone {
    int id = 0;                             // Index of the bone in the final array of bone matrices
    std::string name = "";                  // Name of the bone (matches Assimp node name)
    glm::mat4 offset = glm::mat4(1.0f);     // The inverse bind pose matrix for this bone
    std::vector<Bone> children = {};        // Child bones in the hierarchy
};

// Stores animation keyframe data for a single bone.
struct BoneTransformTrack {
    std::vector<float> positionTimestamps; // Time points for position keyframes
    std::vector<float> rotationTimestamps; // Time points for rotation keyframes
    std::vector<float> scaleTimestamps;    // Time points for scale keyframes

    std::vector<glm::vec3> positions;      // Position keyframe values
    std::vector<glm::quat> rotations;      // Rotation keyframe values
    std::vector<glm::vec3> scales;         // Scale keyframe values
};

// Contains overall animation information, including duration and bone tracks.
struct Animation {
    float duration = 0.0f;              // Total duration of the animation in ticks (from Assimp mDuration)
    float ticksPerSecond = 1.0f;        // Animation ticks per second (from Assimp mTicksPerSecond)
    // Map of bone names to their animation transform tracks
    std::unordered_map<std::string, BoneTransformTrack> boneTransforms;
};

// --- Model and Animation Loading Functions ---

// Recursively reads the bone hierarchy from Assimp's scene graph.
// It populates the 'boneOutput' structure with bone ID, name, offset matrix, and children.
// 'boneInfoTable' maps bone names to their ID and offset matrix.
bool readSkeleton(Bone& boneOutput, aiNode* node,
    const std::unordered_map<std::string, std::pair<int, glm::mat4>>& boneInfoTable) {
    // Check if the current Assimp node corresponds to a bone
    auto it = boneInfoTable.find(node->mName.C_Str());
    if (it != boneInfoTable.end()) { // If node is actually a bone
        boneOutput.name = node->mName.C_Str();
        boneOutput.id = it->second.first;       // Get bone ID
        boneOutput.offset = it->second.second;  // Get bone offset matrix

        // Recursively process children of this bone
        for (unsigned int i = 0; i < node->mNumChildren; i++) {
            Bone child;
            readSkeleton(child, node->mChildren[i], boneInfoTable);
            boneOutput.children.push_back(child);
        }
        return true; // Found and processed a bone
    }
    else { // Current node is not a bone, search in its children
        for (unsigned int i = 0; i < node->mNumChildren; i++) {
            // If a bone is found in any child branch, propagate true up
            if (readSkeleton(boneOutput, node->mChildren[i], boneInfoTable)) {
                return true;
            }
        }
    }
    return false; // No bone found in this branch
}

// Loads mesh data (vertices, indices, bone weights) and builds the skeleton.
void loadModel(const aiScene* scene, aiMesh* mesh,
    std::vector<Vertex>& verticesOutput, std::vector<uint>& indicesOutput,
    Bone& skeletonOutput, uint& nBoneCount) {
    verticesOutput.clear(); // Ensure output vectors are empty before filling
    indicesOutput.clear();

    verticesOutput.reserve(mesh->mNumVertices); // Reserve space to avoid reallocations

    // 1. Load position, normal, UVs for all vertices
    for (unsigned int i = 0; i < mesh->mNumVertices; i++) {
        Vertex vertex;
        // Position
        vertex.position = assimpToGlmVec3(mesh->mVertices[i]);
        // Normal
        vertex.normal = assimpToGlmVec3(mesh->mNormals[i]);
        // UVs (Assimp guarantees at least one UV channel if mTextureCoords[0] is not nullptr)
        if (mesh->mTextureCoords[0]) {
            vertex.uv = glm::vec2(mesh->mTextureCoords[0][i].x, mesh->mTextureCoords[0][i].y);
        }
        else {
            vertex.uv = glm::vec2(0.0f, 0.0f); // Default UV if none exist
        }

        // Initialize bone data
        vertex.boneIds = glm::vec4(0);
        vertex.boneWeights = glm::vec4(0.0f);

        verticesOutput.push_back(vertex);
    }

    // 2. Load bone data (weights and IDs) to vertices
    std::unordered_map<std::string, std::pair<int, glm::mat4>> boneInfoTable;
    std::vector<uint> boneWeightsAssignedCount; // Tracks how many weights have been assigned per vertex
    boneWeightsAssignedCount.resize(verticesOutput.size(), 0);
    nBoneCount = mesh->mNumBones;

    // Loop through each bone in the mesh
    for (uint i = 0; i < nBoneCount; i++) {
        const aiBone* bone = mesh->mBones[i];
        glm::mat4 offsetMatrix = assimpToGlmMatrix(bone->mOffsetMatrix);
        boneInfoTable[bone->mName.C_Str()] = { static_cast<int>(i), offsetMatrix };

        // Loop through each vertex influenced by this bone
        for (unsigned int j = 0; j < bone->mNumWeights; j++) {
            uint vertexId = bone->mWeights[j].mVertexId;
            float weight = bone->mWeights[j].mWeight;

            // Ensure vertexId is within bounds
            if (vertexId >= verticesOutput.size()) {
                std::cerr << "Warning: Bone weight refers to out-of-bounds vertex ID " << vertexId << std::endl;
                continue;
            }

            // Assign bone ID and weight to the next available slot for this vertex
            uint currentBoneCount = boneWeightsAssignedCount[vertexId];
            switch (currentBoneCount) {
            case 0:
                verticesOutput[vertexId].boneIds.x = static_cast<float>(i);
                verticesOutput[vertexId].boneWeights.x = weight;
                break;
            case 1:
                verticesOutput[vertexId].boneIds.y = static_cast<float>(i);
                verticesOutput[vertexId].boneWeights.y = weight;
                break;
            case 2:
                verticesOutput[vertexId].boneIds.z = static_cast<float>(i);
                verticesOutput[vertexId].boneWeights.z = weight;
                break;
            case 3:
                verticesOutput[vertexId].boneIds.w = static_cast<float>(i);
                verticesOutput[vertexId].boneWeights.w = weight;
                break;
            default:
                // More than 4 bones influencing a vertex, ignored for simplicity.
                break;
            }
            boneWeightsAssignedCount[vertexId]++;
        }
    }

    // 3. Normalize bone weights for each vertex to ensure they sum to 1.0
    for (size_t i = 0; i < verticesOutput.size(); i++) {
        glm::vec4& boneWeights = verticesOutput[i].boneWeights;
        float totalWeight = boneWeights.x + boneWeights.y + boneWeights.z + boneWeights.w;
        if (totalWeight > 0.0f) {
            verticesOutput[i].boneWeights /= totalWeight;
        }
    }

    // 4. Load indices (face data)
    indicesOutput.reserve(static_cast<size_t>(mesh->mNumFaces) * 3); // Reserve space for 3 indices per face (assuming triangles)
    for (unsigned int i = 0; i < mesh->mNumFaces; i++) {
        const aiFace& face = mesh->mFaces[i];
        for (unsigned int j = 0; j < face.mNumIndices; j++) {
            indicesOutput.push_back(face.mIndices[j]);
        }
    }

    // 5. Create bone hierarchy (skeleton)
    readSkeleton(skeletonOutput, scene->mRootNode, boneInfoTable);
}

// Loads animation data from the Assimp scene into the Animation structure.
void loadAnimation(const aiScene* scene, Animation& animation) {
    if (scene->mNumAnimations == 0) { // Fix for C4804 warning: directly check if mNumAnimations is zero
        std::cerr << "No animations found in the scene!" << std::endl;
        return;
    }

    // Currently loading only the first animation found in the scene.
    const aiAnimation* anim = scene->mAnimations[0];

    // Set ticks per second. If 0, default to 1.0.
    if (anim->mTicksPerSecond != 0.0) {
        animation.ticksPerSecond = static_cast<float>(anim->mTicksPerSecond);
    }
    else {
        animation.ticksPerSecond = 1.0f;
    }

    // CORRECTED: mDuration from Assimp is already in Ticks.
    // The `animation.duration` member should store this value directly (in ticks).
    // Conversion to seconds for animation playback happens in `main` using `ticksPerSecond`.
    animation.duration = static_cast<float>(anim->mDuration);
    animation.boneTransforms.clear(); // Clear any previous data

    // Load positions, rotations, and scales for each bone's animation channel.
    for (unsigned int i = 0; i < anim->mNumChannels; i++) {
        const aiNodeAnim* channel = anim->mChannels[i];
        BoneTransformTrack track;

        // Load position keyframes
        track.positionTimestamps.reserve(channel->mNumPositionKeys);
        track.positions.reserve(channel->mNumPositionKeys);
        for (unsigned int j = 0; j < channel->mNumPositionKeys; j++) {
            track.positionTimestamps.push_back(static_cast<float>(channel->mPositionKeys[j].mTime));
            track.positions.push_back(assimpToGlmVec3(channel->mPositionKeys[j].mValue));
        }

        // Load rotation keyframes
        track.rotationTimestamps.reserve(channel->mNumRotationKeys);
        track.rotations.reserve(channel->mNumRotationKeys);
        for (unsigned int j = 0; j < channel->mNumRotationKeys; j++) {
            track.rotationTimestamps.push_back(static_cast<float>(channel->mRotationKeys[j].mTime));
            track.rotations.push_back(assimpToGlmQuat(channel->mRotationKeys[j].mValue));
        }

        // Load scale keyframes
        track.scaleTimestamps.reserve(channel->mNumScalingKeys);
        track.scales.reserve(channel->mNumScalingKeys);
        for (unsigned int j = 0; j < channel->mNumScalingKeys; j++) {
            track.scaleTimestamps.push_back(static_cast<float>(channel->mScalingKeys[j].mTime));
            track.scales.push_back(assimpToGlmVec3(channel->mScalingKeys[j].mValue));
        }
        // Store the loaded track using the bone's name
        animation.boneTransforms[channel->mNodeName.C_Str()] = track;
    }
}

// --- OpenGL Buffer Creation and Rendering Helpers ---

// Creates and configures a Vertex Array Object (VAO) for the model.
// Binds vertex attributes (position, normal, UV, bone IDs, bone weights) to buffer data.
unsigned int createVertexArray(const std::vector<Vertex>& vertices, const std::vector<uint>& indices) {
    uint vao = 0, vbo = 0, ebo = 0;

    glGenVertexArrays(1, &vao);
    glGenBuffers(1, &vbo);
    glGenBuffers(1, &ebo);

    glBindVertexArray(vao);

    // Vertex Buffer Object (VBO) for vertex data
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, static_cast<GLsizeiptr>(sizeof(Vertex) * vertices.size()),
        vertices.data(), GL_STATIC_DRAW);

    // Element Buffer Object (EBO) for indices
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, static_cast<GLsizeiptr>(indices.size() * sizeof(uint)),
        indices.data(), GL_STATIC_DRAW);

    // Configure Vertex Attributes
    glEnableVertexAttribArray(0); // Position
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)offsetof(Vertex, position));
    glEnableVertexAttribArray(1); // Normal
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)offsetof(Vertex, normal));
    glEnableVertexAttribArray(2); // UV
    glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)offsetof(Vertex, uv));
    glEnableVertexAttribArray(3); // Bone IDs
    glVertexAttribPointer(3, 4, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)offsetof(Vertex, boneIds));
    glEnableVertexAttribArray(4); // Bone Weights
    glVertexAttribPointer(4, 4, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)offsetof(Vertex, boneWeights));

    glBindVertexArray(0); // Unbind VAO
    return vao;
}

// Loads an image from filepath and creates an OpenGL 2D texture.
unsigned int createTexture(const std::string& filepath) {
    uint textureId = 0;
    int width, height, nrChannels;
    // Load image data with 4 channels (RGBA)
    stbi_set_flip_vertically_on_load(true); // Flip texture vertically for OpenGL
    byte* data = stbi_load(filepath.c_str(), &width, &height, &nrChannels, 4);

    if (!data) {
        std::cerr << "Failed to load texture: " << filepath << std::endl;
        return 0;
    }

    glGenTextures(1, &textureId);
    glBindTexture(GL_TEXTURE_2D, textureId);

    // Set texture wrapping and filtering options
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR); // Use mipmaps
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

    // Upload texture data
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, data);
    glGenerateMipmap(GL_TEXTURE_2D); // Generate mipmaps after uploading data

    stbi_image_free(data); // Free image memory after creating OpenGL texture
    glBindTexture(GL_TEXTURE_2D, 0); // Unbind texture
    return textureId;
}

// --- Animation Playback Logic ---

// Helper function to find the animation segment and interpolation fraction.
// Given a current time `currentTimeTicks` (in animation ticks, normalized to loop)
// and a vector of `times` (keyframe timestamps in ticks),
// it returns the index of the START keyframe (`times[index]`) and the interpolation fraction.
// Returns {last_keyframe_index, 1.0f} if currentTimeTicks is at or beyond the last timestamp.
std::pair<unsigned int, float> getTimeFraction(const std::vector<float>& times, float currentTimeTicks) {
    if (times.empty()) {
        return { 0, 0.0f }; // Should not happen with valid animation data
    }

    // Find the current segment. 'segment' will be the index of the START keyframe.
    // Loop until we find the first keyframe whose timestamp is greater than currentTimeTicks,
    // or we reach the second-to-last keyframe.
    unsigned int segment = 0;
    while (segment < times.size() - 1 && currentTimeTicks >= times[segment + 1]) {
        segment++;
    }

    // If we've reached the last segment (i.e., currentTimeTicks is at or past the last keyframe)
    // then the animation should be clamped to the end pose of the last segment.
    if (segment >= times.size() - 1) {
        // Return the index of the last keyframe, and a fraction of 1.0 (at the end of the segment)
        return { static_cast<unsigned int>(times.size() - 1), 1.0f };
    }

    float startTime = times[segment];
    float endTime = times[segment + 1];

    float duration = endTime - startTime;
    float frac = 0.0f;
    if (duration > 0.0f) {
        frac = (currentTimeTicks - startTime) / duration;
    }

    return { segment, frac }; // Return index of the START keyframe, and fraction
}

// Recursively calculates the bone matrices for the current animation pose.
// 'output' vector will contain the final matrices to be uploaded to the shader.
// 'parentTransform' is the accumulated transform of the parent bone in world space.
// 'globalInverseTransform' is the inverse of the root node's initial transform,
// used to bring the final bone transform into the model's original coordinate system.
void getPose(const Animation& animation, const Bone& skeleton, float currentTimeTicks, // Parameter name clarified
    std::vector<glm::mat4>& output, const glm::mat4& parentTransform,
    const glm::mat4& globalInverseTransform) {

    // Find the animation track for the current bone
    auto it = animation.boneTransforms.find(skeleton.name);

    glm::mat4 localTransform;
    if (it == animation.boneTransforms.end()) {
        // If a bone in the skeleton tree doesn't have an animation track,
        // it means it's a static bone relative to its parent. Its local transform is identity.
        localTransform = glm::mat4(1.0f);
    }
    else {
        const BoneTransformTrack& btt = it->second;

        glm::vec3 finalPosition;
        glm::quat finalRotation;
        glm::vec3 finalScale;

        // Interpolate Position
        if (btt.positions.size() <= 1) { // If 0 or 1 keyframes, just use the first/only one
            finalPosition = btt.positions.empty() ? glm::vec3(0.0f) : btt.positions[0];
        }
        else {
            std::pair<unsigned int, float> fp = getTimeFraction(btt.positionTimestamps, currentTimeTicks);
            unsigned int startKeyIndex = fp.first; // This is the START key index now
            float frac = fp.second;

            // Ensure indices are valid, especially if getTimeFraction returns the last keyframe
            unsigned int endKeyIndex = startKeyIndex + 1;
            if (endKeyIndex >= btt.positions.size()) { // Should be caught by getTimeFraction, but double-check
                endKeyIndex = static_cast<unsigned int>(btt.positions.size() - 1);
                startKeyIndex = endKeyIndex - 1;
                frac = 1.0f; // Clamp to last frame
            }

            glm::vec3 position1 = btt.positions[startKeyIndex];
            glm::vec3 position2 = btt.positions[endKeyIndex];
            finalPosition = glm::mix(position1, position2, frac);
        }

        // Interpolate Rotation
        if (btt.rotations.size() <= 1) {
            finalRotation = btt.rotations.empty() ? glm::quat(1.0f, 0.0f, 0.0f, 0.0f) : btt.rotations[0];
        }
        else {
            std::pair<unsigned int, float> fp = getTimeFraction(btt.rotationTimestamps, currentTimeTicks);
            unsigned int startKeyIndex = fp.first;
            float frac = fp.second;

            unsigned int endKeyIndex = startKeyIndex + 1;
            if (endKeyIndex >= btt.rotations.size()) {
                endKeyIndex = static_cast<unsigned int>(btt.rotations.size() - 1);
                startKeyIndex = endKeyIndex - 1;
                frac = 1.0f;
            }

            glm::quat rotation1 = btt.rotations[startKeyIndex];
            glm::quat rotation2 = btt.rotations[endKeyIndex];
            finalRotation = glm::slerp(rotation1, rotation2, frac);
        }

        // Interpolate Scale
        if (btt.scales.size() <= 1) {
            finalScale = btt.scales.empty() ? glm::vec3(1.0f) : btt.scales[0];
        }
        else {
            std::pair<unsigned int, float> fp = getTimeFraction(btt.scaleTimestamps, currentTimeTicks);
            unsigned int startKeyIndex = fp.first;
            float frac = fp.second;

            unsigned int endKeyIndex = startKeyIndex + 1;
            if (endKeyIndex >= btt.scales.size()) {
                endKeyIndex = static_cast<unsigned int>(btt.scales.size() - 1);
                startKeyIndex = endKeyIndex - 1;
                frac = 1.0f;
            }

            glm::vec3 scale1 = btt.scales[startKeyIndex];
            glm::vec3 scale2 = btt.scales[endKeyIndex];
            finalScale = glm::mix(scale1, scale2, frac);
        }

        glm::mat4 positionMat = glm::translate(glm::mat4(1.0f), finalPosition);
        glm::mat4 rotationMat = glm::toMat4(finalRotation);
        glm::mat4 scaleMat = glm::scale(glm::mat4(1.0f), finalScale);

        localTransform = positionMat * rotationMat * scaleMat;
    }

    glm::mat4 globalTransform = parentTransform * localTransform;

    if (static_cast<size_t>(skeleton.id) < output.size()) {
        output[skeleton.id] = globalInverseTransform * globalTransform * skeleton.offset;
    }
    else {
        std::cerr << "Error: Bone ID " << skeleton.id << " out of bounds for currentPose vector. Max ID: " << output.size() - 1 << std::endl;
    }

    for (const Bone& child : skeleton.children) {
        getPose(animation, child, currentTimeTicks, output, globalTransform, globalInverseTransform);
    }
}

// Global Assimp Importer instance
Assimp::Importer importer;

// --- Main Program ---
int main() {
    // 1. Initialization (Window, OpenGL context)
    int windowWidth = 640;
    int windowHeight = 480;

#ifdef DO_GLFW3
    GLFWwindow* window = glfwInitWindow(windowWidth, windowHeight);
    if (!window) {
        return -1; // Exit if window creation fails
    }
#else
    std::cerr << "Error: No windowing system defined (DO_GLFW3)." << std::endl;
    return -1;
#endif
    // todo: "You must specify your own path for where you would like your resources to be."
    // 2. Model and Texture Paths
    const std::string modelPath = "F:/res/models/man/model.dae";
    const std::string texturePath = "F:/res/models/man/diffuse.png";

    // 3. Load Model and Animation
    const aiScene* scene = importer.ReadFile(modelPath.c_str(),
        aiProcess_Triangulate | aiProcess_FlipUVs | aiProcess_GenSmoothNormals);

    if (!scene || scene->mFlags & AI_SCENE_FLAGS_INCOMPLETE || !scene->mRootNode) {
        std::cerr << "ERROR::ASSIMP::" << importer.GetErrorString() << std::endl;
        glfwDestroyWindow(window);
        glfwTerminate();
        return -1;
    }

    // Assuming the first mesh (index 0) contains the skinned model
    aiMesh* mesh = scene->mMeshes[0];

    std::vector<Vertex> vertices;
    std::vector<uint> indices;
    uint boneCount = 0;
    Animation animation;
    Bone skeleton;

    // The root node's initial transformation matrix from Assimp.
    // This transform defines the model's orientation/position in the scene
    // before any bone animations are applied.
    glm::mat4 globalInverseTransform = glm::inverse(assimpToGlmMatrix(scene->mRootNode->mTransformation));

    loadModel(scene, mesh, vertices, indices, skeleton, boneCount);
    loadAnimation(scene, animation);

    // 4. OpenGL Object Creation (VAO, Texture, Shader)
    uint vao = createVertexArray(vertices, indices);
    uint diffuseTexture = createTexture(texturePath);

    uint shader = createShader(vertexShaderSource, fragmentShaderSource);
    if (shader == 0) {
        glDeleteVertexArrays(1, &vao);
        glDeleteTextures(1, &diffuseTexture);
        glfwDestroyWindow(window);
        glfwTerminate();
        return -1;
    }

    // Get shader uniform locations
    GLint viewProjectionMatrixLocation = glGetUniformLocation(shader, "view_projection_matrix");
    GLint modelMatrixLocation = glGetUniformLocation(shader, "model_matrix");
    GLint boneMatricesLocation = glGetUniformLocation(shader, "bone_transforms");
    GLint textureLocation = glGetUniformLocation(shader, "diff_texture");

    if (viewProjectionMatrixLocation == -1 || modelMatrixLocation == -1 ||
        boneMatricesLocation == -1 || textureLocation == -1) {
        std::cerr << "Warning: One or more shader uniforms not found. Check names." << std::endl;
    }

    // 5. Setup Camera and Model Matrices
    glm::mat4 projectionMatrix = glm::perspective(glm::radians(75.0f),
        static_cast<float>(windowWidth) / windowHeight,
        0.01f, 1000.0f);

    glm::mat4 viewMatrix = glm::lookAt(glm::vec3(0.0f, 0.0f, -15.0f),
        glm::vec3(0.0f, 0.0f, 0.0f),
        glm::vec3(0, 1, 0));

    glm::mat4 viewProjectionMatrix = projectionMatrix * viewMatrix;

    glm::mat4 modelMatrix(1.0f);
    // Apply transformations in S-R-T order (Scale, Rotate, Translate)
    // The glm::op(matrix, ...) functions apply to the matrix on the LEFT,
    // so the operations are effectively applied from RIGHT TO LEFT of the math.
    // Scale first (local to model), then Rotate (local to model), then Translate (world space).
    modelMatrix = glm::scale(modelMatrix, glm::vec3(1.0f)); // Keep scale as is (or adjust if needed)
    // Rotates 180 degrees around the X-axis to flip the model from upside down to upright
    modelMatrix = glm::rotate(modelMatrix, glm::radians(180.0f), glm::vec3(1.0f, 0.0f, 0.0f));
    // ADDED: Rotates 180 degrees around the Y-axis to make the model face the camera
    modelMatrix = glm::rotate(modelMatrix, glm::radians(180.0f), glm::vec3(0.0f, 1.0f, 0.0f));
    modelMatrix = glm::translate(modelMatrix, glm::vec3(0.0f, -0.01f, 0.0f)); // Apply translation for ground alignment

    std::vector<glm::mat4> currentPose(boneCount, glm::mat4(1.0f));

    // Main render loop
    bool isRunning = true;
    while (isRunning) {
        glfwPollEvents();
        if (glfwWindowShouldClose(window)) {
            isRunning = false;
        }

        float elapsedTime = static_cast<float>(glfwGetTime());

        // Calculate the total animation duration in SECONDS
        float animationDurationSeconds = animation.duration / animation.ticksPerSecond;

        // Calculate the current animation time in SECONDS,
        // normalized to loop within the animation's total duration in seconds.
        float loopedTimeSeconds = std::fmod(elapsedTime, animationDurationSeconds);

        // Convert the looped time (in seconds) back to ANIMATION TICKS
        // because the keyframe timestamps in `BoneTransformTrack` are in ticks.
        float loopedTimeTicks = loopedTimeSeconds * animation.ticksPerSecond;

        // Get the current pose for the animation based on the calculated looped time in ticks
        getPose(animation, skeleton, loopedTimeTicks, currentPose, glm::mat4(1.0f), globalInverseTransform);

        // Clear color and depth buffers
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        // Activate shader program and upload uniforms
        glUseProgram(shader);
        glUniformMatrix4fv(viewProjectionMatrixLocation, 1, GL_FALSE, glm::value_ptr(viewProjectionMatrix));
        glUniformMatrix4fv(modelMatrixLocation, 1, GL_FALSE, glm::value_ptr(modelMatrix));
        glUniformMatrix4fv(boneMatricesLocation, static_cast<GLsizei>(currentPose.size()), GL_FALSE, glm::value_ptr(currentPose[0]));

        // Bind texture
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, diffuseTexture);
        glUniform1i(textureLocation, 0);

        // Bind VAO and draw
        glBindVertexArray(vao);
        glDrawElements(GL_TRIANGLES, static_cast<GLsizei>(indices.size()), GL_UNSIGNED_INT, nullptr);

        // Swap buffers
        glfwSwapBuffers(window);
    }

    // 6. Cleanup
    glDeleteProgram(shader);
    glDeleteVertexArrays(1, &vao);
    glDeleteTextures(1, &diffuseTexture);

    glfwDestroyWindow(window);
    glfwTerminate();

    return 0;
}






