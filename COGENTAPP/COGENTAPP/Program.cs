using Microsoft.ML;
using COGENTAPP;

string _trainDataPath = Path.Combine(Environment.CurrentDirectory, "Data", "Extract For Extract for Students.csv.csv");
string _testDataPath = Path.Combine(Environment.CurrentDirectory, "Data", "formatted data testing Data.csv");
string _modelPath = Path.Combine(Environment.CurrentDirectory, "Data", "Model.zip");

MLContext mlContext = new MLContext(seed: 0);

var model = Train(mlContext, _trainDataPath);

string dir = System.IO.Directory.GetCurrentDirectory();
ITransformer Train(MLContext mlContext, string dataPath)
{
    dataPath = "C:/Users/alexm/Documents/COGENTAPP/COGENTAPP/Data/formatted data testing Data.csv";
    IDataView dataView = mlContext.Data.LoadFromTextFile<Product>(dataPath, hasHeader: true, separatorChar: ',');
    var pipeline = mlContext.Transforms.CopyColumns(outputColumnName: "Label", inputColumnName: "SalesQuantity")
.Append(mlContext.Transforms.Categorical.OneHotEncoding(outputColumnName: "ProductIDEncoded", inputColumnName: "ProductID"))
.Append(mlContext.Transforms.Categorical.OneHotEncoding(outputColumnName: "DateEncoded", inputColumnName: "Date"))
.Append(mlContext.Transforms.Categorical.OneHotEncoding(outputColumnName: "StockOnHandEncoded", inputColumnName: "StockOnHand"))
.Append(mlContext.Transforms.Concatenate("Features", "ProductIDEncoded", "DateEncoded", "StockOnHandEncoded", "DeliveredQuantity", "WasteQuantity"))
.Append(mlContext.Regression.Trainers.FastTree());
    var model = pipeline.Fit(dataView);
    return model;

}
void Evaluate(MLContext mlContext, ITransformer model)
{
    Evaluate(mlContext, model);
    IDataView dataView = mlContext.Data.LoadFromTextFile<Product>(_testDataPath, hasHeader: true, separatorChar: ',');
    var predictions = model.Transform(dataView);
    var metrics = mlContext.Regression.Evaluate(predictions, "Label", "Score");
    Console.WriteLine();
    Console.WriteLine($"*************************************************");
    Console.WriteLine($"*       Model quality metrics evaluation         ");
    Console.WriteLine($"*------------------------------------------------");

    Console.WriteLine($"*       RSquared Score:      {metrics.RSquared:0.##}");


}
void TestSinglePrediction(MLContext mlContext, ITransformer model)
{
    TestSinglePrediction(mlContext, model);
    var predictionFunction = mlContext.Model.CreatePredictionEngine<Product, ProductPrediction>(model);
    var testInput = new Product
    {
        ProductID = 75,
        Date = "6/1/2022",
        StockOnHand = 58,
        DeliveredQuantity = 36,
        SalesQuantity = 36,
        WasteQuantity = 6,
        
    };
    var prediction = predictionFunction.Predict(testInput);
    Console.WriteLine($"**********************************************************************");
    Console.WriteLine($"Predicted Quantity: {prediction.SalesQuantity:####}, actual quantity: 36");
    Console.WriteLine($"**********************************************************************");
}

