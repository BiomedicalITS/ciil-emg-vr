using UnityEngine;
using System.Net.Sockets;
using System.Text;
using System;
using System.Collections.Generic;
using System.Linq;

public class TcpClientManagerWithLiveDisplay : MonoBehaviour
{
    public string serverIP;
    public int serverPort;

    public List<int> receivedData; // Public property to access the received data as a list of integers

    private TcpClient client;
    private NetworkStream stream;
    private byte[] buffer = new byte[1024];

    private void Start()
    {
        ConnectToServer();
        InitializeReceivedData();
    }

    private void Update()
    {
        if (stream.DataAvailable)
        {
            int bytesRead = stream.Read(buffer, 0, buffer.Length);
            if (bytesRead > 0)
            {
                string receivedString = Encoding.UTF8.GetString(buffer, 0, bytesRead);
                UpdateReceivedData(receivedString);
            }
        }
    }

    private void InitializeReceivedData()
    {
        receivedData = new List<int>(Enumerable.Repeat(0, 64));
    }

    private void UpdateReceivedData(string receivedDataString)
    {
        string[] values = receivedDataString.Split(new string[] { ";" }, StringSplitOptions.RemoveEmptyEntries);
        Debug.Log(receivedDataString);
        for (int i = 0; i < Mathf.Min(values.Length, receivedData.Count); i++)
        {
            if (int.TryParse(values[i], out int intValue))
            {
                receivedData[i] = intValue;

            }
            else
            {
                Debug.LogError("Failed to parse value: " + values[i]);
            }
        }

        // Now you can access the updated received data using 'receivedData'
        //Debug.Log("Received data from server: " + string.Join(", ", receivedData));
    }

    private void OnDestroy()
    {
        DisconnectFromServer();
    }

    private void ConnectToServer()
    {
        try
        {
            client = new TcpClient();
            client.Connect(serverIP, serverPort);
            stream = client.GetStream();
            Debug.Log("Connected to server.");
        }
        catch (Exception e)
        {
            Debug.Log("Failed to connect to the server: " + e.Message);
        }
    }

    private void DisconnectFromServer()
    {
        if (client != null)
        {
            stream.Close();
            client.Close();
            Debug.Log("Disconnected from the server.");
        }
    }

    public void SendMessageToServer(string message)
    {
        try
        {
            byte[] data = Encoding.UTF8.GetBytes(message);
            stream.Write(data, 0, data.Length);
            stream.Flush();
            Debug.Log("Sent message to the server: " + message);
        }
        catch (Exception e)
        {
            Debug.LogError("Error sending data: " + e.Message);
        }
    }
}
