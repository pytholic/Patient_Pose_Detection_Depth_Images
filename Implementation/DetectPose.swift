import CoreML
import Foundation

private class DetectPatientPoseInput: MLFeatureProvider {
    var featureNames: Set<String> = []

    private let inputFeatureName: String
    private var image = UIImage()
    private var imageData: UnsafeMutablePointer<Float32>?

    init(image: UIImage,
         inputFeatuerName: String)
    {
        self.image = image
        inputFeatureName = inputFeatuerName
        featureNames.insert(inputFeatuerName)
    }

    deinit {
        imageData?.deallocate()
    }

    func featureValue(for featureName: String) -> MLFeatureValue? {
        if featureName == inputFeatureName {
            return MLFeatureValue(pixelBuffer: image.pixelBuffer()!)
        }
        return nil
    }

    func normalize(_ image: UIImage) -> UIImage {
        let width = 320
        let height = 240
        let bytesPerComponent = 4
        let channel = 1
        let imageSize = width * height * bytesPerComponent

        guard Int(image.size.width) == width,
              Int(image.size.height) == height,
              image.cgImage?.bitsPerComponent == 8,
              image.cgImage?.bitsPerPixel == 8
        else { // setting include width, height type is uint8 and 1 channel
            fatalError("\(#function) : you should fixed image setting")
        }

        let cgimage = image.cgImage
        guard let originalData = cgimage?.dataProvider?.data,
              let original = CFDataGetBytePtr(originalData)
        else {
            fatalError("\(#function) : get original buffer failed.")
        }

        imageData = UnsafeMutablePointer<Float32>.allocate(capacity: width * height)
        guard let imageData = imageData else { fatalError("\(#function) : unknwon error") }

        // 0 ~ 255 (uint8) -> -1.0 ~ 1.0 (float32)
        let normValue: Float32 = 127.5
        for w in 0..<width {
            for h in 0..<height {
                imageData[h * width + w] = (Float32(original[h * width + w]) - normValue) / normValue
            }
        }

        guard let provider = CGDataProvider(data: Data(bytesNoCopy: imageData,
                                                       count: imageSize,
                                                       deallocator: .none) as CFData)
        else {
            fatalError("\(#function) : provider load failed.")
        }

        guard let cgImage = CGImage(width: width,
                                    height: height,
                                    bitsPerComponent: bytesPerComponent * 8,
                                    bitsPerPixel: bytesPerComponent * channel * 8,
                                    bytesPerRow: bytesPerComponent * channel * width,
                                    space: CGColorSpaceCreateDeviceGray(),
                                    bitmapInfo: CGBitmapInfo(rawValue: 0),
                                    provider: provider,
                                    decode: nil,
                                    shouldInterpolate: false,
                                    intent: .defaultIntent)
        else {
            fatalError("\(#function) : create cgimage failed by unknown error")
        }

        return UIImage(cgImage: cgImage)
    }
}

private class DetectPatientPoseOutput: MLFeatureProvider {
    var featureNames: Set<String> = []

    private let provider: MLFeatureProvider
    private let outputFeatureName: String

    init(featueres: MLFeatureProvider,
         outputFeatureName: String)
    {
        provider = featueres
        self.outputFeatureName = outputFeatureName
        featureNames.insert(outputFeatureName)
    }

    func featureValue(for featureName: String) -> MLFeatureValue? {
        if featureName == outputFeatureName {
            return provider.featureValue(for: featureName)
        }
        return nil
    }

    func patientPose() -> PatientPosePreset {
        guard let result = featureValue(for: outputFeatureName)?.multiArrayValue else {
            fatalError("\(#function): unknown error check mlmodel...")
        }

        let pointer = result.dataPointer.assumingMemoryBound(to: Float32.self)
        let values: [Float32] = [pointer[0], pointer[1], pointer[2], pointer[3]]

        if let max = values.max(),
           let index = values.firstIndex(of: max)
        {
            if index == 0 { return .supineHeadLeft }
            else if index == 1 { return .supineHeadRight }
            else if index == 3 { return .standing }
        }
        return .unknown // unknown at 2 or other
    }
}

public class DetectPatientPoseModel {
    let model: MLModel

    let modelName = "detect_patient_pose"
    let inputFeatureName = "input_1"
    let outputFeatureName = "647"

    public init() {
        guard let url = Bundle(for: Self.self)
            .url(forResource: modelName,
                 withExtension: "mlmodelc"),
            let model = try? MLModel(contentsOf: url)
        else {
            fatalError("\(#function) : DetectHeadModel MLModel Load Failed...")
        }

        self.model = model
    }

    public func prediction(image: UIImage) -> PatientPosePreset {
        let input = DetectPatientPoseInput(image: image,
                                           inputFeatuerName: inputFeatureName)
        guard let result = try? model.prediction(from: input,
                                                 options: MLPredictionOptions())
        else {
            fatalError("\(#function) : failed")
        }

        return DetectPatientPoseOutput(featueres: result,
                                       outputFeatureName: outputFeatureName).patientPose()
    }
}
