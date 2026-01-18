#!/usr/bin/env python
"""Script de inicialização do serviço Steam Game Success Predictor."""

from steam.app import app

if __name__ == '__main__':
    app.run(debug=True)
