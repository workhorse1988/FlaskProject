from flask_sqlalchemy import SQLAlchemy
from datetime import datetime

db = SQLAlchemy()

class Analysis(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)
    filename = db.Column(db.String(255), nullable=False)
    flow_rate = db.Column(db.Float, nullable=True)
    diameter = db.Column(db.Float, nullable=True)
    slope = db.Column(db.Float, nullable=True)
    coefficient = db.Column(db.Float, nullable=True)
    results = db.Column(db.JSON, nullable=True)

    def to_dict(self):
        return {
            'id': self.id,
            'timestamp': self.timestamp.isoformat(),
            'filename': self.filename,
            'flow_rate': self.flow_rate,
            'diameter': self.diameter,
            'slope': self.slope,
            'coefficient': self.coefficient,
            'results': self.results
        }