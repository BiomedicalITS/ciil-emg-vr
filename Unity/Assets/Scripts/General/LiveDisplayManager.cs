using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;

public class LiveDisplayManager : MonoBehaviour
{
    // Array to store references to the pixel objects
    public Image[] pixelImages;

    // Reference to the TcpClientManagerWithLiveDisplay
    public TcpClientManager tcpClientManager;

    // Start is called before the first frame update
    void Start()
    {
        // Initialize the array with pixel images
        InitializePixelImages();
    }

    // Update is called once per frame
    void Update()
    {
        // Update pixel colors based on received data
        UpdatePixelColors();
    }

    // Initialize the pixelImages array with references to the child pixel images
    public void InitializePixelImages()
    {
        pixelImages = new Image[64];

        for (int i = 0; i < 64; i++)
        {
            string pixelName = "pixel" + i.ToString("D2"); // Ensure two-digit formatting
            Transform pixel = transform.Find(pixelName);

            if (pixel != null)
            {
                Image image = pixel.GetComponent<Image>();
                if (image != null)
                {
                    pixelImages[i] = image;
                }
            }
        }
    }

    // Update the color of each pixel based on the received data
    private void UpdatePixelColors()
    {
        if (tcpClientManager != null && tcpClientManager.receivedData != null)
        {
            for (int i = 0; i < tcpClientManager.receivedData.Count; i++)
            {
                if (i < pixelImages.Length)
                {
                    Image image = pixelImages[i];

                    if (image != null)
                    {
                        // Map the received data value to a color between purple and yellow
                        float t = Mathf.InverseLerp(0, 1200, tcpClientManager.receivedData[i]);
                        Color color = Color.Lerp(Color.blue, Color.yellow, t);

                        // Set the color of the pixel
                        image.color = color;
                    }
                }
            }
        }
    }
}