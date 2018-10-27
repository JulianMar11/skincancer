//
//  ViewControllerCamera.swift
//  Skintect
//
//  Created by Matthias Bitzer on 26.10.18.
//  Copyright © 2018 Matthias Bitzer. All rights reserved.
//

import UIKit
import AVFoundation


class ViewControllerCamera: UIViewController, AVCapturePhotoCaptureDelegate ,passForward{
 
    

    var captureSession = AVCaptureSession()
    var backCamera : AVCaptureDevice?
    var frontCamera : AVCaptureDevice?
    var currentCamera : AVCaptureDevice?
    
    var photoOutput : AVCapturePhotoOutput?
    
    var cameraPreviewLayer : AVCaptureVideoPreviewLayer?
    var image : UIImage?
    var delegate : passForward?
    
    override func viewDidLoad() {
        super.viewDidLoad()

        setupCaptureSession()
        setupDevice()
        setupInputOutput()
        setupPreviewLayer()
        startRunningCaptureSession()
        
    }
    
    func setupCaptureSession(){
        captureSession.sessionPreset = AVCaptureSession.Preset.photo
    }
    
    func setupDevice(){
        let deviceDiscoverySession = AVCaptureDevice.DiscoverySession(deviceTypes: [AVCaptureDevice.DeviceType.builtInWideAngleCamera], mediaType: AVMediaType.video, position: AVCaptureDevice.Position.unspecified)
        let devices = deviceDiscoverySession.devices
        
        for device in devices {
            
            if device.position == AVCaptureDevice.Position.back{
                backCamera = device
            }else if device.position == AVCaptureDevice.Position.front{
                frontCamera = device
            }
            
        }
        currentCamera = backCamera
    }
    
    func setupInputOutput(){
        do{
            let captureDeviceInput = try AVCaptureDeviceInput(device: currentCamera!)
            captureSession.addInput(captureDeviceInput)
            photoOutput = AVCapturePhotoOutput()
            photoOutput?.setPreparedPhotoSettingsArray([AVCapturePhotoSettings(format:[AVVideoCodecKey: AVVideoCodecType.jpeg])], completionHandler: nil)
            captureSession.addOutput(photoOutput!)
        } catch {
            print("Error")
        }
        
    }
    
    func setupPreviewLayer(){
        cameraPreviewLayer = AVCaptureVideoPreviewLayer(session: captureSession)
        cameraPreviewLayer?.videoGravity = AVLayerVideoGravity.resizeAspectFill
        cameraPreviewLayer?.connection?.videoOrientation = AVCaptureVideoOrientation.portrait
        cameraPreviewLayer?.frame = self.view.frame
        self.view.layer.insertSublayer(cameraPreviewLayer!, at: 0)
        
        
    }
    
    func startRunningCaptureSession(){
        captureSession.startRunning()
        
    }
    

    @IBAction func cameraButtonPressed(_ sender: Any) {
        let settings = AVCapturePhotoSettings()
        photoOutput?.capturePhoto(with: settings, delegate: self)
    }
    
    override func prepare(for segue: UIStoryboardSegue, sender: Any?) {
        if(segue.identifier == "saveIm"){
            let dest = segue.destination as! ViewControllerCameraPreview
            dest.image = self.image
            dest.delegate=self
            
        }
    }
    
    func photoOutput(_ output: AVCapturePhotoOutput, didFinishProcessingPhoto photo: AVCapturePhoto, error: Error?) {
        if let imageData = photo.fileDataRepresentation(){
            print(imageData)
            self.image = UIImage(data: imageData)
            performSegue(withIdentifier: "saveIm", sender: self)
        }
    }
    
    func getImage(image: UIImage) {
        self.dismiss(animated: false) {
            self.delegate?.getImage(image: image)
        }
        
    }
    

}
