using UnityEngine;
using TMPro;

public class MatchPosition : MonoBehaviour
{
    // Reference to the public GameObject
    public Transform targetObject;

    // Reference to the Text component on the canvas
    private TextMeshProUGUI textComponent;

    // Start is called before the first frame update
    void Start()
    {
        // Get the Text component on the canvas
        textComponent = GetComponentInChildren<TextMeshProUGUI>();

        // Check if the Text component is found
        if (textComponent == null)
        {
            Debug.LogError("Text component not found on the canvas!");
        }
    }

    // Update is called once per frame
    void Update()
    {
        // Check if the targetObject is assigned
        if (targetObject != null)
        {
            // Match the position of the canvas to the targetObject
            textComponent.transform.position = targetObject.position;

            // Find the EMGRawReader GameObject in the scene
            GameObject emgObject = GameObject.Find("EMGRawReader"); // Replace with the actual name of your EMGRawReader GameObject

            // Access the EMG_accuracy directly from the EMGRawReader GameObject
            if (emgObject != null)
            {
                textComponent.text = emgObject.GetComponent<EMGRawReader>().EMG_velocity.ToString("F2");
            }
            else
            {
                Debug.LogWarning("EMGRawReader GameObject not found in the scene!");
            }
        }
        else
        {
            Debug.LogWarning("Target Object is not assigned in the inspector!");
        }
    }
}
