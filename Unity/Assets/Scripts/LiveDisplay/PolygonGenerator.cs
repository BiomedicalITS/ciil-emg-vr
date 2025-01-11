using UnityEngine;

public class PolygonGenerator : MonoBehaviour
{
    public GameObject cubePrefab; // Prefab for the cubes that will form the polygon
    private int sides = 16; // Number of sides for the polygon
    public float radius = 5f; // Radius of the polygon
    private int numStages = 4; // Number of stages to generate
    private int[] channelMap = {10, 22, 12, 24, 13, 26, 7, 28, 1, 30, 59, 32, 53, 34, 48, 36,
                                62, 16, 14, 21, 11, 27, 5, 33, 63, 39, 57, 45, 51, 44, 50, 40,
                                8, 18, 15, 19, 9, 25, 3, 31, 61, 37, 55, 43, 49, 46, 52, 38,
                                6, 20, 4, 17, 2, 23, 0, 29, 60, 35, 58, 41, 56, 47, 54, 42};
    private int cubeCounter = 0; // Counter for cube numbering

    void Start()
    {
        // Start polygon generation on initialization
        GeneratePolygons();
    }

    void GeneratePolygons()
    {
        // Calculate the Y-axis offset for positioning each stage
        Vector3 yOffset = Vector3.up * CalculateYOffset();

        // Generate each stage of the polygon
        for (int i = 0; i < numStages; i++)
        {
            GeneratePolygon(i * yOffset);
        }
    }

    void GeneratePolygon(Vector3 offset)
    {
        // Find the GameObject that will act as the parent container
        GameObject liveDisplayGenerator = GameObject.Find("LiveDisplayGenerator");

        if (liveDisplayGenerator == null)
        {
            Debug.LogError("LiveDisplayGenerator GameObject not found.");
            return;
        }

        // Create a new container for the polygon
        GameObject polygonContainer = new GameObject("PolygonContainer");
        polygonContainer.transform.parent = liveDisplayGenerator.transform;

        // Calculate the angle increment for positioning each cube
        float angleIncrement = 360f / sides;

        // Generate each side of the polygon
        for (int i = 0; i < sides; i++)
        {
            float angle = i * angleIncrement;
            float x = radius * Mathf.Cos(Mathf.Deg2Rad * angle);
            float z = radius * Mathf.Sin(Mathf.Deg2Rad * angle);

            Vector3 position = new Vector3(x, 0f, z) + offset;

            // Instantiate the cube at the calculated position
            GameObject cube = Instantiate(cubePrefab, position, Quaternion.identity, polygonContainer.transform);

            // Calculate and apply the scale factor based on the radius and number of sides
            float scaleFactor = 2f * Mathf.Tan(Mathf.PI / sides) * radius;
            cube.transform.localScale = new Vector3(scaleFactor, scaleFactor, 0.001f);

            // Orient the cube to face the center
            cube.transform.LookAt(polygonContainer.transform.position + offset);

            // Name the cube according to the channel map
            int cubeNumber = channelMap[cubeCounter++];
            cube.name = "Cube" + cubeNumber.ToString("D2");
        }

        // Position the container at the center with the offset
        polygonContainer.transform.position = liveDisplayGenerator.transform.position;
    }

    float CalculateYOffset()
    {
        // Calculate the vertical offset for each stage
        float angleIncrement = 360f / sides;
        float scaleFactor = 2f * Mathf.Tan(Mathf.PI / sides) * radius;
        return scaleFactor;
    }
}
