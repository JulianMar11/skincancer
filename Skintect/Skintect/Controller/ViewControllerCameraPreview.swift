//
//  ViewControllerCameraPreview.swift
//  Skintect
//
//  Created by Matthias Bitzer on 26.10.18.
//  Copyright © 2018 Matthias Bitzer. All rights reserved.
//

import UIKit

protocol passForward {
    func getImage(image : UIImage)
}

class ViewControllerCameraPreview: UIViewController {
    
    var image : UIImage?
    var delegate : passForward?
    
    @IBOutlet weak var imageView: UIImageView!
    override func viewDidLoad() {
        super.viewDidLoad()
        imageView.image = image
        // Do any additional setup after loading the view.
    }
    
    @IBAction func retakeButtonPressed(_ sender: Any) {
        self.dismiss(animated: true, completion: nil)
    }
    
    @IBAction func usePhotoPressed(_ sender: Any) {
        self.dismiss(animated: false) {
            self.delegate?.getImage(image: self.image!)
        }
    }
    /*
    // MARK: - Navigation

    // In a storyboard-based application, you will often want to do a little preparation before navigation
    override func prepare(for segue: UIStoryboardSegue, sender: Any?) {
        // Get the new view controller using segue.destination.
        // Pass the selected object to the new view controller.
    }
    */

}
