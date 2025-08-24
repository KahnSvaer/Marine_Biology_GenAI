using UnityEngine;
using UnityGLTF;
using UnityGLTF.Loader;
using System.Threading.Tasks;
using System.IO;
using UnityEngine.Networking;
using System.IO.Compression;

[RequireComponent(typeof(ServerStatusChecker))]
public class RuntimeGLTFUnzipAndApply : MonoBehaviour
{
    string glbZipUrl = "";

    void Start()
    {
        string glbZipUrlServer = GetComponent<ServerStatusChecker>().serverUrl;
        glbZipUrl = glbZipUrlServer + "get-3d-file";
        if (string.IsNullOrEmpty(glbZipUrlServer))
        {
            Debug.LogError("GLB ZIP URL is empty. Please set it in the ServerStatusChecker component.");
            return;
        }
    }

    private string unzipFolder => Path.Combine(Application.persistentDataPath, "UnzippedGLB");

    public async void DownloadAndSpawnModelWithTexture()
    {
        await DownloadUnzipAndInstantiate();
    }

    async Task DownloadUnzipAndInstantiate()
    {
        Debug.Log("Starting download and unzip process...");
        string zipPath = Path.Combine(Application.persistentDataPath, "model.zip");
        Debug.Log("Downloading zip: " + glbZipUrl);

        // 1. Download ZIP
        UnityWebRequest www = UnityWebRequest.Get(glbZipUrl);
        www.downloadHandler = new DownloadHandlerBuffer();
        await www.SendWebRequest();

#if UNITY_2020_1_OR_NEWER
        if (www.result != UnityWebRequest.Result.Success)
#else
        if (www.isNetworkError || www.isHttpError)
#endif
        {
            Debug.LogError("ZIP download failed: " + www.error);
            return;
        }

        File.WriteAllBytes(zipPath, www.downloadHandler.data);
        Debug.Log("ZIP saved at: " + zipPath);

        // 2. Unzip
        if (Directory.Exists(unzipFolder)) Directory.Delete(unzipFolder, true);
        ZipFile.ExtractToDirectory(zipPath, unzipFolder);
        Debug.Log("Unzipped to: " + unzipFolder);

        // 3. Locate GLB & Texture
        string glbPath = Directory.GetFiles(unzipFolder, "*.glb", SearchOption.AllDirectories)[0];
        string texturePath = Directory.GetFiles(unzipFolder, "*.png", SearchOption.AllDirectories)[0];

        // 4. Instantiate GLB at origin
        await LoadGLBAtOrigin(glbPath, texturePath);
    }

    async Task LoadGLBAtOrigin(string glbPath, string texturePath)
    {
        var loader = new FileLoader(Path.GetDirectoryName(glbPath));
        AsyncCoroutineHelper coroutineHelper = gameObject.AddComponent<AsyncCoroutineHelper>();

        var importOptions = new ImportOptions
        {
            DataLoader = loader,
            AsyncCoroutineHelper = coroutineHelper
        };

        string fileName = Path.GetFileName(glbPath);
        Transform glbRoot = new GameObject("GLB_Instance").transform;
        glbRoot.position = Vector3.zero;

        var gltfImporter = new GLTFSceneImporter(fileName, importOptions)
        {
            SceneParent = glbRoot
        };

        try
        {
            await gltfImporter.LoadSceneAsync();
            Debug.Log("GLB loaded.");

            // 5. Load texture from PNG
            Texture2D loadedTexture = LoadTextureFromFile(texturePath);
            if (loadedTexture == null)
            {
                Debug.LogError("Failed to load texture.");
                return;
            }

            // 6. Create material
            Material newMat = new Material(Shader.Find("Universal Render Pipeline/Lit"));
            newMat.mainTexture = loadedTexture;

            // 7. Apply material to all renderers in GLB
            foreach (Renderer r in glbRoot.GetComponentsInChildren<Renderer>())
            {
                r.material = newMat;
            }

            Debug.Log("Material applied to GLB.");
        }
        catch (System.Exception e)
        {
            Debug.LogError("Error loading GLB: " + e.Message);
        }
    }

    Texture2D LoadTextureFromFile(string texturePath)
    {
        if (!File.Exists(texturePath))
        {
            Debug.LogError("Texture file not found: " + texturePath);
            return null;
        }

        byte[] fileData = File.ReadAllBytes(texturePath);
        Texture2D tex = new Texture2D(2, 2);
        if (tex.LoadImage(fileData))
        {
            return tex;
        }

        return null;
    }
}
