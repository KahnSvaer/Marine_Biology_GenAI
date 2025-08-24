using UnityEngine;
using UnityEngine.Rendering.Universal;

public class RuntimeTextureTesting : MonoBehaviour
{
    [Header("Assign in Inspector")]
    public GameObject prefabToSpawn;     // Prefab to spawn at runtime
    public Texture2D textureToApply;     // Texture to apply on the spawned object

    void Start()
    {
        if (prefabToSpawn == null || textureToApply == null)
        {
            Debug.LogWarning("Prefab or texture is not assigned.");
            return;
        }

        // 1. Spawn the prefab
        GameObject spawnedObject = Instantiate(prefabToSpawn, transform.position, Quaternion.identity);

        // 2. Get the renderer
        Renderer renderer = spawnedObject.GetComponent<Renderer>();
        if (renderer == null)
        {
            Debug.LogWarning("Spawned object does not have a Renderer.");
            return;
        }

        // 3. Create a URP Lit material
        Material urpLitMaterial = new Material(Shader.Find("Universal Render Pipeline/Lit"));

        if (urpLitMaterial == null)
        {
            Debug.LogError("URP Lit shader not found. Make sure URP is correctly set up.");
            return;
        }

        // 4. Apply the texture to the Base Map
        urpLitMaterial.SetTexture("_BaseMap", textureToApply);

        // 5. Assign the new material
        renderer.material = urpLitMaterial;

        Debug.Log("Spawned object with URP Lit material and applied texture.");
    }
}
