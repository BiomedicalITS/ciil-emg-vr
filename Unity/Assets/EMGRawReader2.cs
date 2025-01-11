using System;
using System.Globalization;
using System.Net;
using System.Net.Sockets;
using System.Text;
using System.Threading;
using UnityEngine;

public class EMGRawReader2 : MonoBehaviour
{
    private UdpClient udpClient;
    private Thread receiveThread;
    private bool isReceiving = true;

    private readonly object dataLock = new object();
    private string receivedData;

    public int prediction;
    public float accuracy;
    public double timestamp;

    void Start()
    {
        udpClient = new UdpClient(12346);
        receiveThread = new Thread(new ThreadStart(ReceiveData));
        receiveThread.IsBackground = true;
        receiveThread.Start();
    }

    void Update()
    {
        lock (dataLock)
        {
            if (!string.IsNullOrEmpty(receivedData))
            {
                ParseData(receivedData);
                receivedData = null;
            }
        }
    }

    private void ReceiveData()
    {
        try
        {
            IPAddress ipAddress = IPAddress.Parse("127.0.0.1");
            IPEndPoint remoteEndPoint = new IPEndPoint(ipAddress, 12346);

            while (isReceiving)
            {
                byte[] receiveBytes = udpClient.Receive(ref remoteEndPoint);
                string receiveString = Encoding.UTF8.GetString(receiveBytes);
                //Debug.Log($"Received data: {receiveString}");

                lock (dataLock)
                {
                    receivedData = receiveString;
                }
            }
        }
        catch (Exception e)
        {
            Debug.LogError($"ReceiveData error: {e}");
        }
    }

    private void ParseData(string data)
    {
        try
        {
            string[] parts = data.Split(' ');
            if (parts.Length != 3)
            {
                Debug.LogError($"ParseData error: Expected 3 parts but got {parts.Length} in data: {data}");
                return;
            }

            if (!int.TryParse(parts[0], out prediction))
            {
                Debug.LogError($"ParseData error: Invalid prediction value in data: {data}");
                return;
            }

            if (!float.TryParse(parts[1], NumberStyles.Float, CultureInfo.InvariantCulture, out accuracy))
            {
                Debug.LogError($"ParseData error: Invalid accuracy value in data: {data}");
                return;
            }

            if (!double.TryParse(parts[2], NumberStyles.Float, CultureInfo.InvariantCulture, out timestamp))
            {
                Debug.LogError($"ParseData error: Invalid timestamp value in data: {data}");
                return;
            }
        }
        catch (Exception e)
        {
            Debug.LogError($"ParseData error: {e}");
        }
    }

    void OnApplicationQuit()
    {
        isReceiving = false;

        if (udpClient != null)
        {
            udpClient.Close();
        }

        if (receiveThread != null && receiveThread.IsAlive)
        {
            receiveThread.Join();
        }
    }
}
