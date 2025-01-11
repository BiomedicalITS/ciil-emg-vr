using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System.Globalization;
using System;
using System.Threading;
using System.Net;
using System.Net.Sockets;
using System.Text;

public class VTFeedback : MonoBehaviour
{
    private string serverIP = "127.0.0.1";
    private int serverPort = 12348;

    Thread readThread; // Thread for reading data
    private TcpClient client; // TCP client for server connection
    private NetworkStream stream; // Network stream for data communication

    private static VTFeedback playerInstance;

    void Awake()
    {
        // Ensure this object persists across scene loads and implement singleton pattern
        DontDestroyOnLoad(this);

        if (playerInstance == null)
        {
            playerInstance = this;
        }
        else
        {
            Destroy(gameObject);
        }
        ConnectToServer();
    }

    private void ConnectToServer()
    {
        // Attempt to connect to the server
        try
        {
            client = new TcpClient();
            client.Connect(serverIP, serverPort);
            stream = client.GetStream();
            Debug.Log("Connected to server.");
            Write(false);
        }
        catch (Exception e)
        {
            Debug.Log("Failed to connect to the server: " + e.Message);
        }
    }

    private void OnDestroy()
    {
        // Disconnect from the server when the object is destroyed
        DisconnectFromServer();
    }

    private void DisconnectFromServer()
    {
        // Close the stream and client connection
        if (client != null)
        {
            stream.Close();
            client.Close();
            Debug.Log("Disconnected from server.");
        }
    }

    public void Write(bool _isOn)
    {
        // Send a message to the server
        try
        {
            string message = _isOn ? "1" : "0";
            byte[] data = Encoding.UTF8.GetBytes(message);
            stream.Write(data, 0, data.Length);
            stream.Flush();
            Debug.Log("Sent message to server: " + message);
        }
        catch (Exception e)
        {
            Debug.LogError("Error sending data: " + e.Message);
        }
    }
}
