"use strict";
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
const Route_1 = __importDefault(global[Symbol.for('ioc.use')]("Adonis/Core/Route"));
Route_1.default.get('/', 'SubmissionsController.index');
Route_1.default.get('/processingSubmission', 'SubmissionsController.checkCompletion');
Route_1.default.post('/submitFile', 'SubmissionsController.submit');
Route_1.default.get('/document/:filename/:pageNumber?', 'DocumentsController.viewPage');
//# sourceMappingURL=routes.js.map