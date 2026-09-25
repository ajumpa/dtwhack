# Pi camera relay

The Pi runs a Wi-Fi hotspot and relays the ESP32 camera (USB, `/dev/ttyACM0`) as MJPEG over HTTP.
The Pi does no image processing; detection and display happen on the laptop.

## 1. Hotspot (once)

The Pi runs Ubuntu Server, where netplan/systemd-networkd manage the network and cannot run an
access point. NetworkManager takes over `wlan0` only; `eth0` stays with netplan.

Do this over an Ethernet session (e.g. a laptop/desktop sharing its connection on the cable):
the Pi leaves its current Wi-Fi in step 4, and an SSH session over Wi-Fi drops there.

```bash
# 1. Back up; make eth0 persistent (fixed 10.43.0.2 + DHCP, so the cable always works)
sudo cp -r /etc/netplan ~/netplan-backup
sudo tee /etc/netplan/40-eth0.yaml > /dev/null <<'YAML'
network:
  version: 2
  ethernets:
    eth0:
      optional: true
      dhcp4: true
      addresses: [10.43.0.2/24]
YAML
sudo chmod 600 /etc/netplan/40-eth0.yaml
sudo netplan apply

# 2. NetworkManager; stop cloud-init rewriting netplan at boot
sudo apt update && sudo apt install network-manager dnsmasq-base
echo 'network: {config: disabled}' | sudo tee /etc/cloud/cloud.cfg.d/99-disable-network-config.cfg

# 3. NetworkManager manages wlan0 and nothing else
sudo tee /etc/NetworkManager/conf.d/99-weedbot.conf > /dev/null <<'CONF'
[keyfile]
unmanaged-devices=*,except:interface-name:wlan0
CONF
sudo systemctl restart NetworkManager

# 4. Remove home Wi-Fi from netplan (the Pi leaves that network here)
sudo mv /etc/netplan/50-cloud-init.yaml ~/
sudo netplan apply

# 5. Hotspot; proto rsn forces WPA2 (default here was WPA1, which clients reject)
sudo nmcli connection add type wifi ifname wlan0 con-name Hotspot autoconnect yes ssid weedbot mode ap 802-11-wireless.band bg ipv4.method shared wifi-sec.key-mgmt wpa-psk wifi-sec.psk 'ajumpahotspot' wifi-sec.proto rsn wifi-sec.pairwise ccmp wifi-sec.group ccmp
sudo nmcli connection up Hotspot
ip -br addr show wlan0    # 10.42.0.1/24
```

The Pi is `10.42.0.1` on `weedbot`, wherever it is; no outside Wi-Fi is involved. Clients on the hotspot
get internet only if the Pi has it (e.g. through the Ethernet cable).

If the 2.4 GHz band is crowded: `sudo nmcli connection modify Hotspot 802-11-wireless.channel 11 && sudo nmcli connection up Hotspot`

Undo: `sudo nmcli connection delete Hotspot && sudo cp ~/netplan-backup/*.yaml /etc/netplan/ && sudo netplan apply`

## 2. Install the relay as a service (once)

From the project root on the laptop:

```bash
scp pi/pi_server.py pi/weedbot-camera.service ajumpa@10.42.0.1:~
```

On the Pi:

```bash
sudo mkdir -p /opt/weedbot
sudo mv ~/pi_server.py /opt/weedbot/
sudo mv ~/weedbot-camera.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now weedbot-camera
journalctl -u weedbot-camera -f     # "reading /dev/ttyACM0", then fps every 10 s
```

To run it by hand instead: `python3 pi_server.py` (stop the service first; only one program can read the port).

## 3. View on the laptop

Join the `weedbot` Wi-Fi (password `ajumpahotspot`), then open <http://10.42.0.1:8000/>.

| URL | Content |
|---|---|
| `/` | page showing the live stream |
| `/stream` | MJPEG stream, also readable with `cv2.VideoCapture("http://10.42.0.1:8000/stream")` |
| `/frame.jpg` | latest single frame |

## Updating

```bash
scp pi/pi_server.py ajumpa@10.42.0.1:~ && ssh ajumpa@10.42.0.1 'sudo mv ~/pi_server.py /opt/weedbot/ && sudo systemctl restart weedbot-camera'
```
