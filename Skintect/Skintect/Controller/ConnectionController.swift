//
//  ConnectionController.swift
//  Skintect
//
//  Created by Matthias Bitzer on 26.10.18.
//  Copyright © 2018 Matthias Bitzer. All rights reserved.
//

import Foundation
import UIKit
import Alamofire
protocol ConnectionDelegate {
    func requestFinished(result : String)
}


class ConnectionController{
    
    var delegate : ConnectionDelegate?
    
    func upLoadImage(image : UIImage){
        
        
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
                            self.delegate?.requestFinished(result: response.result.value!)
                            
                        } else {
                            
                        }                    }
                case .failure( _):
                    print("Fehler")
                }
        }
        )
        
    }
    
    
}
