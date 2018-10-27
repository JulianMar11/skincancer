//
//  ViewControllerSecondScreen.swift
//  Skintect
//
//  Created by Matthias Bitzer on 26.10.18.
//  Copyright © 2018 Matthias Bitzer. All rights reserved.
//

import UIKit
import Alamofire

class ViewControllerSecondScreen: UIViewController,UIImagePickerControllerDelegate, UINavigationControllerDelegate, passForward, ConnectionDelegate {

    

    
    let connect = ConnectionController()
      var imageHolder = UIImage()
    override func viewDidLoad() {
        super.viewDidLoad()
        connect.delegate=self

        // Do any additional setup after loading the view.
    }
    
    @IBAction func buttonPressed(_ sender: Any) {
        
        performSegue(withIdentifier: "camera", sender: self)
       /* if UIImagePickerController.isSourceTypeAvailable(.camera){
            
            let imagePicker = UIImagePickerController()
            imagePicker.delegate=self
            imagePicker.sourceType = .camera
            imagePicker.allowsEditing = false
            self.present(imagePicker, animated: true ,completion: nil)
            
            
        }*/
        
    }
    
    
    
    @IBAction func homeButtonPressed(_ sender: Any) {
        self.dismiss(animated: true, completion: nil)
    }
    /*func imagePickerController(_ picker: UIImagePickerController, didFinishPickingMediaWithInfo info: [UIImagePickerController.InfoKey : Any]) {
        
        let image = info[UIImagePickerController.InfoKey.originalImage] as! UIImage
        imageHolder = image
        
        
        
        let url = try! URLRequest(url: URL(string:"http://172.20.10.3/predictImage")!, method: .post, headers: nil)
        
        Alamofire.upload(
            multipartFormData: { (form: MultipartFormData) in
                let im = image.jpegData(compressionQuality: 0.5)
                form.append(im!, withName:"image", fileName: "file.png", mimeType: "image/png")
                // multipartFormData.append(Data("Hallo"), withName: "image", fileName: "file.png", mimeType: "image/png")
        },
            with: url,
            encodingCompletion: { encodingResult in
                switch encodingResult {
                case .success(let upload, _, _):
                    print("Erfolg")
                    
                    upload.responseString { response in
                        if((response.result.value) != nil) {
                            print(response.result.value)
                            print(response.request)
                        } else {
                            
                        }                    }
                case .failure( _):
                    print("Fehler")
                }
        }
        )
        
        
        
        
        
        picker.dismiss(animated: true, completion: {
            self.performSegue(withIdentifier: "showResults", sender: self)
        })
        
        
    }*/
    
    func getImage(image: UIImage) {
        connect.upLoadImage(image: image)
    }
    
    override func prepare(for segue: UIStoryboardSegue, sender: Any?) {
        if segue.identifier == "camera"{
            let dest = segue.destination as! ViewControllerCamera
            dest.delegate = self
            
            
        }
    }
    
    func requestFinished(result : String){
        print(result)
        print("Erfolgreich")
        
    }
    

}
