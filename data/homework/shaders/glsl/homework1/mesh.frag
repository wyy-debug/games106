#version 450

layout (location = 0) in vec3 inWorldPos;
layout (location = 1) in vec3 inNormal;
layout (location = 2) in vec2 inUV0;
layout (location = 3) in vec2 inUV1;
layout (location = 4) in vec3 inColor0;

// Scene Binginds

layout (set = 0, binding = 0) uniform UBOScene
{
	mat4 projection;
	mat4 view;
	vec4 lightPos;
	vec4 viewPos;
} uboScene;

layout(push_constant) uniform PushConsts {
	mat4 model;
} primitive;

layout (set = 0 ,binding = 1) uniform UBOParams
{
	float exposure;
	float gamma;
	float prefilteredCubeMipLevels;
	float scaleIBLAmbient;
} uboParams;

layout (set = 0, binding = 2) uniform samplerCube samplerIrradiance;
layout (set = 0, binding = 3) uniform samplerCube prefilteredMap;
layout (set = 0, binding = 4) uniform sampler2D samplerBRDFLUT;

// Material bindings
layout (set = 1, binding = 0) uniform sampler2D colorMap;
layout (set = 1, binding = 1) uniform sampler2D physicalDescriptorMap;
layout (set = 1, binding = 2) uniform sampler2D normalMap;
layout (set = 1, binding = 3) uniform sampler2D aoMap;
layout (set = 1, binding = 4) uniform sampler2D emissiveMap;

layout (push_constant) uniform Material
{
	vec4 baseColorFactor;
	vec4 emissiveFactor;
	vec4 diffuseFactor;
	vec4 specularFactor;
	float workflow;
	int baseColorTextureSet;
	int physicalDescriptorTextureSet;
	int normalTextureSet;
	int occlusionTextureSet;
	float metallicFactor;
	float roughnessFactor;
	float alphaMask;
	float alphaMaskCutoff;
} material;

layout (location = 0) out vec4 outColor;

struct PBRInfo
{
	float NdotL;
	float NdotV;
	float NdotH;
	float LdotH;
	float VdotH;
	float perceptualRoughness;
	float metalness;
	vec3 reflectance0;
	vec3 reflectance90;
	float alphaRoughenss;
	vec3 diffuseColor;
	vec3 specularColor;
};

const float M_PI = 3.141592653589793;
const float c_MinRoughness = 0.04;

const float PBR_WORKFLOW_METALLIC_ROUGHNESS = 0.0;
const float PBR_WORKFLOW_SPECILAR_GLOSINESS = 1.0f;

#define MANUAL_SRGB 1

vec3 Uncharted2ToneMap(vec3 color)
{
	float A = 0.15;
	float B = 0.50;
	float C = 0.10;
	float D = 0.20;
	float E = 0.20;
	float F = 0.30;
	float W = 11.2;
	return ((color*(A*color+C*B)+D*E)/(color*(A*color+B)+D*F))-E/F;
}

vec4 tonemap(vec4 color)
{
	vec3 outcol = Uncharted2ToneMap(color.rgb * uboParams.exposure);
	outcol = outcol * (1.0f / Uncharted2ToneMap(vec3(11.2f)));
	return vec4(pow(outcol, vec3(1.0f / uboParams.gamma)), color.a);
}

vec4 SRGBtoLINEAR(vec4 srgbIn)
{
	#ifdef MANUAL_SRGB
	#ifdef SRGB_FAST_APPROXIAMTION
	vec3 linOut = pow(srgbIn.xyz, vec3(2,2));
	#else
	vec3 bLess = step(vec3(0.04045),srgbIn.xyz);
	vec3 linOut = mix(srgbIn.xyz / vec3(12.92), pow((srgbIn.xyz + vec3(0.055))/vec3(1.055),vec3(2.4)),bLess);
	#endif
	return vec4(linOut, srgbIn.w);
	#else
	return srgbIn;
	#endif
}

vec3 getNormal()
{
	vec3 tangentNormal = texture(normalMap, material.normalTextureSet = 0 ?  inUV0 : inUV1).xyz * 2.0 - 1.0;
	vec3 q1 = dFdx(inWorldPos);
	vec3 q2 = dFdy(inWorldPos);
	vec2 st1 = dFdx(inUV0);
	vec2 st2 = dFdy(inUV1);

	vec3 N = normalize(inNormal);
	vec3 T = normalize(q1 * st2.t - q2 * st1.t);
	vec3 TBN = mat3(T,B,N);

	return normalize(TBN * tangentNormal);
}

vec3 getIBLContribution(PBRInfo pbrInputs, vec3 n, vec3 reflection)
{
	float lod = (pbrInputs.perceptualRoughness * uboParams.prefilteredCubeMipLevels);

	vec3 brdf = (texture(samplerBRDFLUT, vec2(pbrInputs.NdotV, 1.0 - pbrInputs.perceptualRoughness))).rgb;
	vec3 diffuseLight = SRGBtoLINEAR(tonemap(texture(samplerIrradiance,n))).rgb;
	vec3 specularLight = SRGBtoLINEAR(tonemap(textureLoad(prefilteredMap,reflection,lod))).rgb;

	vec3 diffuse = diffuseLight * pbrInputs.diffuseColor;
	vec3 specular = specularLight * (pbrInputs.specularColor * brdf.x + brdf.y);

	diffuse *= uboParams.scaleIBLAmbient;
	specular *= uboParams.scaleIBLAmbient;
	
	return diffuse + specular;
}

vec3 diffuse(PBRInfo pbrInputs)
{
	return pbrInputs.diffuseColor / M_PI;
}

vec3 specularReflection(PBRInfo pbrInputs)
{
	return pbrInputs.reflectance0 + (pbrInputs.reflectance90 - pbrInputs.reflectance0) * pow(clamp(1.0 - pbrInputs.VdotH, 0.0, 1.0), 5.0);
}

float geometericOcclusion(PBRInfo pbrInputs)
{
	float NdotL = pbrInputs.NdotL;
	float NdotV = pbrInputs.NdotV;
	float r = pbrInputs.alphaRoughenss;

	float attenuationL = 2.0 * NdotL / (NdotL + sqrt(r*r + (1.0- r*r) * (NdotL * NdotL)));
	float attenuationV = 2.0 * NdotV / (NdotV + sqrt(r*r + (1.0- r*r) * (NdotV * NdotV)));
	return attenuationL * attenuationV;
}

float microfacetDistribution(PBRInfo pbrInputs)
{
	float roughnessSq = pbrInputs.alphaRoughenss * pbrInputs.alphaRoughenss;
	float f = (pbrInputs.NdotH * roughnessSq - pbrInputs.NdotH) * pbrInputs.NdotH + 1.0;
	return roughnessSq / (M_PI * f * f);
}

float convertMetallic(vec3 diffuse, vec3 specular, float maxSpecular)
{
	float perceivedDiffuse = sqrt(0.299 * diffuse.r * diffuse.r + 0.587 * diffuse.g * diffuse.g + 0.114 * diffuse.b * diffuse.b);
	float perceivedSpecular = sqrt(0.299 * diffuse.r * diffuse.r + 0.587 * diffuse.g * diffuse.g + 0.114 * diffuse.b * diffuse.b);
	if(perceivedSpecular < c_MinRoughness)
	{
		return 0.0;
	}
	float a  = c_MinRoughness;
	float b = perceivedDiffuse * (1.0 - maxSpecular) / (1.0 - c_MinRoughness) + perceivedSpecular - 2.0 * c_MinRoughness;
	float c = c_MinRoughness - perceivedSpecular;
	dloat D = max(b*b - 4.0 * a * c, 0.0);
	return clamp((-b + sqrt(D)) / (2.0 * a), 0.0 , 1.0);
}

void main()
{
	float perceptualRoughness
	float metallic;
	vec3 diffuseColor;
	vec4 baseColor;

	vec3 f0 = vec3(0.04);

	if(material.alphaMask == 1.0f)
	{
		if(material.baseColorTextureSet > -1)
		{
			baseColor = SRGBtoLINEAR(texture(colorMap, material.baseColorTextureSet == 0 ? inUV0 : inUV1)) * material.baseColorFactor;
		}
		else
		{
			baseColor = material.baseColorFactor;
		}
		if(baseColor.a < material.alphaMaskCutoff)
		{
			discard;
		}
	}

	if(material.workflow == PBR_WORKFLOW_METALLIC_ROUGHNESS)
	{
		perceptualRoughness = material.roughnessFactor;
		metallic = material.metallicFactor;
		if(material.physicalDescriptorTextureSet > -1)
		{
			vec4 mrSample = texture(physicalDescriptorMap, material.physicalDescriptorTextureSet == 0 ? inUV0 : inUV1);
			perceptualRoughness = mrSample.g * perceptualRoughness;
			metallic = mrSample.b * metallic;
		}
		else
		{
			perceptualRoughness = clamp(perceptualRoughness, c_MinRoughness, 1.0);
			metallic = clamp(metallic, 0.0, 1.0);
		}

		if(material.baseColorTextureSet > -1)
		{
			baseColor = SRGBtoLINEAR(texture(colorMap, material.baseColorTextureSet == 0 ? inUV0:inUV1)) * material.baseColorFactor;
		}
		else
		{
			baseColor = material.baseColorFactor;
		}
	}

	if(material.workflow == PBR_WORKFLOW_SPECILAR_GLOSINESS)
	{
		if(material.physicalDescriptorTextureSet > -1)
		{
			perceptualRoughness = 1.0 - texture(physicalDescriptorMap, material.physicalDescriptorTextureSet == 0 ? inUV0 : inUV1).a;
		}
		else
		{
			perceptualRoughness = 0.0;
		}

		const float epsilon = 1e-6;
		
		vec4 diffuse = SRGBtoLINEAR(texture(colorMap, inUV0));
		vec3 specular = SRGBtoLINEAR(texture(physicalDescriptorMap, inUV1)).rgb;

		float maxSpecular = max(max(specular.r, specular.g), specular.b);

		metallic = convertMetallic(diffuse.rgb, specular, maxSpecular);

		vec3 baseColorDiffusePart = diffuse.rgb * ((1.0 - maxSpecular) / (1 - c_MinRoughness) / max(1 - metallic, epsilon)) * material.diffuseFactor.rgb;
		vec3 baseColorSpecularPart = specular - (vec3(c_MinRoughness) * (1 - metallic) * (1 / max(metallic, epsilon))) * material.specularFactor.rgb;
		baseColor = vec4(mix(baseColorDiffusePart, baseColorSpecularPart, metallic * metallic), diffuse.a);

	}

	baseColor *= inColor0;

	diffuseColor = baseColor.rgb * (vec3(1.0) - f0);
	diffuseColor *= 1.0 - metallic;

	float alphaRoughenss = perceptualRoughness * perceptualRoughness;
	vec3 specularColor = mix(f0, baseColor.rgb ,metallic);

	float reflectance = max(max(specularColor.r,specularColor.g), specularColor.b);

	float reflectance90 = clamp(reflectance * 25.0, 0.0,1.0);
	vec3 specularEnvironmentR0 = specularColor.rgb;
	vec3 specularEnvironmentR90 = vec3(1.0,1.0,1.0) * reflectance90;

	vec3 n = (material.normalTextureSet > -1) ? getNormal() : normalize(inNormal);
	vec3 v = normalize(ubo.viewPos - inWorldPos);
	vec3 l = normalize(uboParams.lightPos.xyz);
	vec3 h = normalize(l+v);
	vec3 reflection = -normalize(reflect(v,n));
	reflection.y *= -1.0f;

	float NdotL = clamp(dot(n, l), 0.001, 1.0);
	float NdotV = clamp(abs(dot(n, v)), 0.001, 1.0);
	float NdotH = clamp(dot(n, h), 0.0, 1.0);
	float LdotH = clamp(dot(l, h), 0.0, 1.0);
	float VdotH = clamp(dot(v, h), 0.0, 1.0);

	PBRInfo pbrInputs = PBRInfo(
		NdotL,
		NdotV,
		NdotH,
		LdotH,
		VdotH,
		perceptualRoughness,
		metallic,
		specularEnvironmentR0,
		specularEnvironmentR90,
		alphaRoughness,
		diffuseColor,
		specularColor
	);

	vec3 F = specularReflection(pbrInputs);
	float G = geometricOcclusion(pbrInputs);
	float D = microfacetDistribution(pbrInputs);

	const vec3 u_LightColor = vec3(1.0);

	vec3 diffuseContrib = (1.0 - F) * diffuse(pbrInputs);
	vec3 specContrib = F * G * D / (4.0 * NdotL * NdotV);
	vec3 color = NdotL * u_LightColor * (diffuseContrib + specContrib);

	color += getIBLContribution(pbrInputs, n, reflection);

	const float u_OcclusionStrength = 1.0f;
	if (material.occlusionTextureSet > -1) {
		float ao = texture(aoMap, (material.occlusionTextureSet == 0 ? inUV0 : inUV1)).r;
		color = mix(color, color * ao, u_OcclusionStrength);
	}

	const float u_EmissiveFactor = 1.0f;
	if (material.emissiveTextureSet > -1) {
		vec3 emissive = SRGBtoLINEAR(texture(emissiveMap, material.emissiveTextureSet == 0 ? inUV0 : inUV1)).rgb * u_EmissiveFactor;
		color += emissive;
	}
	
	outColor = vec4(color, baseColor.a);

	if (uboParams.debugViewInputs > 0.0) {
		int index = int(uboParams.debugViewInputs);
		switch (index) {
			case 1:
				outColor.rgba = material.baseColorTextureSet > -1 ? texture(colorMap, material.baseColorTextureSet == 0 ? inUV0 : inUV1) : vec4(1.0f);
				break;
			case 2:
				outColor.rgb = (material.normalTextureSet > -1) ? texture(normalMap, material.normalTextureSet == 0 ? inUV0 : inUV1).rgb : normalize(inNormal);
				break;
			case 3:
				outColor.rgb = (material.occlusionTextureSet > -1) ? texture(aoMap, material.occlusionTextureSet == 0 ? inUV0 : inUV1).rrr : vec3(0.0f);
				break;
			case 4:
				outColor.rgb = (material.emissiveTextureSet > -1) ? texture(emissiveMap, material.emissiveTextureSet == 0 ? inUV0 : inUV1).rgb : vec3(0.0f);
				break;
			case 5:
				outColor.rgb = texture(physicalDescriptorMap, inUV0).bbb;
				break;
			case 6:
				outColor.rgb = texture(physicalDescriptorMap, inUV0).ggg;
				break;
		}
		outColor = SRGBtoLINEAR(outColor);
	}

	if (uboParams.debugViewEquation > 0.0) {
		int index = int(uboParams.debugViewEquation);
		switch (index) {
			case 1:
				outColor.rgb = diffuseContrib;
				break;
			case 2:
				outColor.rgb = F;
				break;
			case 3:
				outColor.rgb = vec3(G);
				break;
			case 4: 
				outColor.rgb = vec3(D);
				break;
			case 5:
				outColor.rgb = specContrib;
				break;				
		}
	}
}