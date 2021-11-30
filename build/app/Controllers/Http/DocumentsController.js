"use strict";
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
const Output_1 = __importDefault(global[Symbol.for('ioc.use')]("App/Models/Output"));
class DocumentsController {
    async viewPage(ctx) {
        const fileName = ctx.request.param('filename');
        if (fileName === null) {
            return ctx.response.notFound();
        }
        const pageNumber = Number.parseInt(ctx.request.param('pageNumber', '1'));
        const out = await Output_1.default.query().where('file_name', fileName).andWhere('page_number', pageNumber).first();
        if (out === null) {
            return ctx.response.notFound();
        }
        const edition = await Output_1.default.query().where('file_name', fileName);
        if (edition === null) {
            return ctx.response.notFound();
        }
        const pageCount = edition.map((value) => value.pageNumber).reduce((prev, curr) => Math.max(prev, curr));
        const previousPage = pageNumber > 0 ? pageNumber - 1 : null;
        const nextPage = pageNumber < pageCount ? pageNumber + 1 : null;
        return ctx.view.render('document', { fileName, pageNumber, previousPage, nextPage });
    }
}
exports.default = DocumentsController;
//# sourceMappingURL=DocumentsController.js.map