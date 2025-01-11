using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System.Globalization;
using System;
using System.Threading;
using System.Net;
using System.Net.Sockets;
using System.Text;
using System.Linq;

public class EMGRawReader : MonoBehaviour
{
    private readonly string IP = "127.0.0.1";
    // Port to receive EMG predictions from LibEMG
    private readonly int port = 12347;
    // Port for Unity to send Context (GrabSetUp2.cs)
    private readonly int server_port = 12350;

    public string timestamp;

    public int EMG_prediction;
    public float EMG_velocity;
    public bool switchScene = true;
    private bool isReadingData = false;

    private PositionSaver positionSaver;

    private void Start()
    {
        pose_sender = new UdpClient(IP, server_port);

        GameObject _objects = GameObject.Find("test");
        positionSaver = _objects.GetComponent<PositionSaver>();
    }

    Thread readThread; // Thread for reading UDP messages
    UdpClient client; // UDP client for receiving data
    UdpClient pose_sender; // UDP client for sending data
    readonly IPEndPoint serverTarget; // Target endpoint for sending data

    private static EMGRawReader playerInstance;

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
    }

    void Update()
    {
        // Check if switchScene is true and data reading hasn't started yet
        if (switchScene && !isReadingData)
        {
            StartReadingData();
            isReadingData = true;
        }
    }

    public void StartReadingData()
    {
        // Create and start a background thread for receiving UDP messages
        readThread = new Thread(new ThreadStart(ReceiveData))
        {
            IsBackground = true
        };
        readThread.Start();
        Write("READY");
        IPEndPoint anyIP = new(IPAddress.Any, 0);
        byte[] buff = pose_sender.Receive(ref anyIP);
        string path = Encoding.UTF8.GetString(buff); // Decode bytes to string
        positionSaver.SavePath = path;
    }

    void OnApplicationQuit()
    {
        // Stop the thread and close UDP clients on application quit
        Write("Q");
        StopThread();
    }

    public void StopThread()
    {
        // Abort the reading thread and close UDP clients
        try
        {
            if (readThread.IsAlive)
            {
                readThread.Abort();
            }
        }
        catch (Exception e)
        {
            Debug.LogError(e);
        }

        pose_sender.Close();
        client.Close();
    }

    private void ReceiveData()
    {
        // Function to receive data in the background thread
        client = new UdpClient(port);
        IPEndPoint anyIP = new(IPAddress.Any, 0);
        string[] pid_to_name = { "Chuck", "Closed", "Open", "Index Point", "Pinch", "Neutral", "Wrist Extension", "Wrist Flexion" };
        while (true)
        {
            try
            {
                byte[] buff = client.Receive(ref anyIP); // Receive bytes from any IP

                string text = Encoding.UTF8.GetString(buff); // Decode bytes to string
                string[] splitData = text.Split(' '); // Split received data

                int pred = int.Parse(splitData[0]);
                int[] pred_to_pose = { 3, 2, 8, 5, 1, 0, 7, 6, 0 }; // see README

                EMG_prediction = pred_to_pose[pred];
                EMG_velocity = 1.0f;
                timestamp = splitData[1];

                Debug.Log(timestamp + " VrCID: " + EMG_prediction + " (" + pid_to_name[pred] + ")");

            }
            catch (Exception err)
            {
                Debug.Log(err.ToString()); // Log any errors
            }
        }
    }

    public void Write(string strMessage)
    {
        // Function to send a message via UDP
        byte[] arr = Encoding.UTF8.GetBytes(strMessage);
        pose_sender.Send(arr, arr.Length, serverTarget);
    }

    // Method to be called by other script to change switchScene and start reading data
    public void SetSwitchScene(bool value)
    {
        switchScene = value;
    }
}
