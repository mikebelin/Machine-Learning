"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
var csv_dataset_1 = require("./datasets/csv_dataset");
var url_data_source_1 = require("./sources/url_data_source");
function csv(source, csvConfig) {
    if (csvConfig === void 0) { csvConfig = {}; }
    return new csv_dataset_1.CSVDataset(new url_data_source_1.URLDataSource(source), csvConfig);
}
exports.csv = csv;
//# sourceMappingURL=readers.js.map