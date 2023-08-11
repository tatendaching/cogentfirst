using Microsoft.ML.Data;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;


namespace COGENTAPP
{
    public class Product
    {
        [LoadColumn(0)]
        public int ProductID { get; set; }

        [LoadColumn(1)]
        public string? Date { get; set; }

        [LoadColumn(2)]
        public float StockOnHand { get; set; }

        [LoadColumn(3)]
        public float DeliveredQuantity { get; set; }

        [LoadColumn(4)]
        public float SalesQuantity { get; set; }

        [LoadColumn(5)]
        public float WasteQuantity { get; set; }
    }
    public class ProductPrediction
    {
        [ColumnName("Score")]
        public float SalesQuantity;


    }

}
